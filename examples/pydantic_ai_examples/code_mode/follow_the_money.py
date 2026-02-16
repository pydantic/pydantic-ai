"""CodeMode Example: Fraud Ring Detection via Transaction Graph Traversal.

This example demonstrates where code mode doesn't just reduce roundtrips — it makes
the task qualitatively easier to solve correctly. The scenario: given a flagged account,
trace money flows through a layered transaction network, identify convergence points
(accounts receiving funds from multiple upstream sources), and rank suspects.

The API is deliberately decomposed and hostile to manual traversal:

  1. Paginated results (3 per page) — the LLM must loop to get the full picture.
  2. Multi-currency transactions requiring FX rate lookups and arithmetic.
  3. Batch wire transfers that hide real recipients behind an extra API call.
  4. Verbose records with many irrelevant fields per response.

With traditional tool calling, the LLM must:
- Make 3-4 round-trips per account just to paginate through transactions
- Call get_exchange_rate + mentally multiply for every foreign-currency transfer
- Notice that "batch" transactions need expansion, then call get_batch_details
- Mentally track visited accounts, running totals, and source sets across ~100 calls
- All of this correctly across 3 hops, ~15 accounts, and ~45 transactions

With code mode, the LLM writes a BFS loop with a while-page loop inside, does
`amount * rate` inline, expands batches in a for-loop, and returns a summary.

Run:
    uv run -m pydantic_ai_examples.code_mode.follow_the_money
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import logfire

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic_ai.run import AgentRunResult
from pydantic_ai.runtime.monty import MontyRuntime
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.code_mode import CodeModeToolset

# =============================================================================
# Configuration
# =============================================================================

AMOUNT_THRESHOLD = 5000
PAGE_SIZE = 3

PROMPT = """\
Starting from account ACC-001, which has been flagged for suspicious activity, \
trace all outgoing money flows up to 3 hops deep. Only follow transactions of \
$5,000 or more (after converting to USD using the exchange rate tool for any \
non-USD transactions) — smaller amounts are likely legitimate business expenses.

Some transactions are batch wire transfers that bundle payments to multiple \
recipients. You MUST expand these by calling the batch details tool — the \
transaction list and transaction detail endpoints only show the batch total, \
not where the money actually went.

Build the full transaction network and identify convergence points — accounts \
that receive money from 2 or more different upstream accounts in the traced \
network. These suggest layering. Flag all convergence points and report their \
names, number of upstream sources, and total inflows (converted to USD).\
"""

MODEL = 'gateway/anthropic:claude-sonnet-4-5'
MAX_RETRIES = 5

# =============================================================================
# Mock Transaction Network
# =============================================================================

# Account names — used by the transaction builder and ground truth
_ACCOUNT_NAMES: dict[str, str] = {
    'ACC-001': 'Viktor Petrov',
    'ACC-002': 'Oceanic Trading LLC',
    'ACC-003': 'Baltic Shipping Co',
    'ACC-004': 'Maria Santos',
    'ACC-005': 'Sunrise Consulting',
    'ACC-006': 'Digital Media Partners',
    'ACC-007': 'Northern Logistics',
    'ACC-008': 'Quick Cash Services',
    'ACC-009': 'Golden Gate Holdings',
    'ACC-010': 'Harbor Real Estate',
    'ACC-011': 'Cafe Milano',
    'ACC-012': 'Island Getaway Travel',
    'ACC-013': 'City Newsstand',
    'ACC-014': 'Metro Dry Cleaning',
    'ACC-015': 'Pine Street Deli',
}

_EXTERNAL_NAMES: dict[str, str] = {
    'EXT-101': 'Pacific Rim Imports',
    'EXT-102': 'Nordic Freight Alliance',
    'EXT-103': 'Apex Strategy Group',
    'EXT-104': 'Broadview Media Inc',
    'EXT-105': 'Continental Warehousing',
    'EXT-106': 'Premier Currency Exchange',
    'EXT-107': 'Westfield Capital Partners',
    'EXT-108': 'Santos Family Trust',
    'EXT-109': 'Eastern European Trade Consortium',
    'EXT-110': 'Meridian Advisory Services',
    'EXT-111': 'Bay Area Venture Fund',
    'EXT-112': 'Sierra Investment Corp',
    'EXT-113': 'Creative Solutions Agency',
    'EXT-114': 'TransAtlantic Commodities',
    'EXT-115': 'Caspian Energy Group',
    'EXT-116': 'Silk Road Textiles',
    'EXT-117': 'Danube River Freight',
    'EXT-118': 'Coral Bay Exports',
    'EXT-119': 'Summit Peak Advisors',
    'EXT-120': 'Black Sea Minerals',
    'EXT-121': 'Fjord Line Cargo',
    'EXT-122': 'Redwood Strategies LLC',
    'EXT-123': 'Amber Coast Holdings',
}

_ALL_NAMES: dict[str, str] = {**_ACCOUNT_NAMES, **_EXTERNAL_NAMES}

# Verbose account records — realistic financial API response shape.
# The LLM only needs id/name for the task, but gets 15-20 fields per call.
_ACCOUNTS: dict[str, dict[str, Any]] = {
    'ACC-001': {
        'id': 'ACC-001',
        'name': 'Viktor Petrov',
        'type': 'individual',
        'status': 'flagged',
        'date_of_birth': '1978-06-14',
        'nationality': 'Russian Federation',
        'address': '42 Nevsky Prospect, Apt 15, St. Petersburg, Russia 191025',
        'phone': '+7-812-555-0147',
        'email': 'v.petrov@petrovconsulting.com',
        'occupation': 'Import/Export Consultant',
        'employer': 'Self-employed',
        'annual_income': '$180,000-$250,000',
        'tax_id': 'XX-XXX7823',
        'risk_score': 78,
        'kyc_status': 'review_pending',
        'last_review_date': '2024-01-05',
        'account_opened': '2020-03-15',
        'account_balance': 12450.00,
    },
    'ACC-002': {
        'id': 'ACC-002',
        'name': 'Oceanic Trading LLC',
        'type': 'business',
        'status': 'active',
        'registration_number': 'LLC-2018-33847',
        'jurisdiction': 'Delaware',
        'incorporation_date': '2018-07-22',
        'registered_agent': 'National Corporate Services',
        'address': '1200 Harbor Blvd, Suite 400, Wilmington, DE 19801',
        'phone': '+1-302-555-0182',
        'email': 'accounts@oceanictrading.com',
        'industry': 'International Trade',
        'annual_revenue_band': '$5M-$10M',
        'employee_count': 28,
        'primary_contact': 'Elena Vasquez, CFO',
        'tax_id': 'XX-XXX4190',
        'risk_score': 45,
        'kyc_status': 'current',
        'last_review_date': '2023-09-18',
        'account_opened': '2018-08-01',
        'account_balance': 234780.50,
    },
    'ACC-003': {
        'id': 'ACC-003',
        'name': 'Baltic Shipping Co',
        'type': 'business',
        'status': 'active',
        'registration_number': 'LLC-2017-28934',
        'jurisdiction': 'New York',
        'incorporation_date': '2017-03-10',
        'registered_agent': 'Harbor Legal Services',
        'address': '45 Water Street, 12th Floor, New York, NY 10004',
        'phone': '+1-212-555-0293',
        'email': 'finance@balticshipping.com',
        'industry': 'Maritime Logistics',
        'annual_revenue_band': '$10M-$25M',
        'employee_count': 65,
        'primary_contact': 'Andrei Volkov, Director of Operations',
        'tax_id': 'XX-XXX8821',
        'risk_score': 38,
        'kyc_status': 'current',
        'last_review_date': '2023-11-02',
        'account_opened': '2017-04-15',
        'account_balance': 567200.00,
    },
    'ACC-004': {
        'id': 'ACC-004',
        'name': 'Maria Santos',
        'type': 'individual',
        'status': 'active',
        'date_of_birth': '1985-11-28',
        'nationality': 'Brazilian',
        'address': '789 Brickell Ave, Apt 2201, Miami, FL 33131',
        'phone': '+1-305-555-0174',
        'email': 'maria.santos@gmail.com',
        'occupation': 'Real Estate Agent',
        'employer': 'Luxe Properties International',
        'annual_income': '$95,000-$150,000',
        'tax_id': 'XX-XXX5567',
        'risk_score': 32,
        'kyc_status': 'current',
        'last_review_date': '2023-12-15',
        'account_opened': '2021-06-20',
        'account_balance': 45890.25,
    },
    'ACC-005': {
        'id': 'ACC-005',
        'name': 'Sunrise Consulting',
        'type': 'business',
        'status': 'active',
        'registration_number': 'LLC-2019-41205',
        'jurisdiction': 'Nevada',
        'incorporation_date': '2019-01-08',
        'registered_agent': 'Silver State Filings',
        'address': '3960 Howard Hughes Pkwy, Suite 500, Las Vegas, NV 89169',
        'phone': '+1-702-555-0138',
        'email': 'billing@sunriseconsulting.net',
        'industry': 'Management Consulting',
        'annual_revenue_band': '$1M-$5M',
        'employee_count': 8,
        'primary_contact': 'David Park, Managing Partner',
        'tax_id': 'XX-XXX9034',
        'risk_score': 51,
        'kyc_status': 'current',
        'last_review_date': '2023-10-30',
        'account_opened': '2019-02-14',
        'account_balance': 189340.00,
    },
    'ACC-006': {
        'id': 'ACC-006',
        'name': 'Digital Media Partners',
        'type': 'business',
        'status': 'active',
        'registration_number': 'LLC-2020-55672',
        'jurisdiction': 'California',
        'incorporation_date': '2020-05-14',
        'registered_agent': 'Pacific Registered Agents',
        'address': '2049 Century Park East, Suite 2500, Los Angeles, CA 90067',
        'phone': '+1-310-555-0216',
        'email': 'ap@digitalmediapartners.co',
        'industry': 'Digital Marketing',
        'annual_revenue_band': '$1M-$5M',
        'employee_count': 15,
        'primary_contact': 'Sarah Kim, Head of Finance',
        'tax_id': 'XX-XXX2288',
        'risk_score': 41,
        'kyc_status': 'current',
        'last_review_date': '2023-08-22',
        'account_opened': '2020-06-01',
        'account_balance': 112450.75,
    },
    'ACC-007': {
        'id': 'ACC-007',
        'name': 'Northern Logistics',
        'type': 'business',
        'status': 'active',
        'registration_number': 'LLC-2016-19483',
        'jurisdiction': 'Illinois',
        'incorporation_date': '2016-09-30',
        'registered_agent': 'Midwest Corporate Services',
        'address': '233 S Wacker Drive, Suite 8400, Chicago, IL 60606',
        'phone': '+1-312-555-0187',
        'email': 'invoices@northernlogistics.com',
        'industry': 'Freight & Logistics',
        'annual_revenue_band': '$5M-$10M',
        'employee_count': 42,
        'primary_contact': 'Michael Torres, VP Finance',
        'tax_id': 'XX-XXX6145',
        'risk_score': 29,
        'kyc_status': 'current',
        'last_review_date': '2023-07-14',
        'account_opened': '2016-10-15',
        'account_balance': 423100.00,
    },
    'ACC-008': {
        'id': 'ACC-008',
        'name': 'Quick Cash Services',
        'type': 'business',
        'status': 'active',
        'registration_number': 'MSB-2019-08841',
        'jurisdiction': 'Florida',
        'incorporation_date': '2019-11-05',
        'registered_agent': 'Sunshine State Filings',
        'address': '100 SE 2nd Street, Suite 3200, Miami, FL 33131',
        'phone': '+1-786-555-0144',
        'email': 'compliance@quickcashsvc.com',
        'industry': 'Money Services Business',
        'annual_revenue_band': '$1M-$5M',
        'employee_count': 11,
        'primary_contact': 'Roberto Diaz, Compliance Officer',
        'tax_id': 'XX-XXX7790',
        'risk_score': 67,
        'kyc_status': 'current',
        'last_review_date': '2024-01-02',
        'account_opened': '2019-12-01',
        'account_balance': 78920.50,
    },
    'ACC-009': {
        'id': 'ACC-009',
        'name': 'Golden Gate Holdings',
        'type': 'business',
        'status': 'active',
        'registration_number': 'LLC-2019-45892',
        'jurisdiction': 'Delaware',
        'incorporation_date': '2019-03-15',
        'registered_agent': 'National Corporate Services',
        'address': '1455 Market Street, Suite 600, San Francisco, CA 94103',
        'phone': '+1-415-555-0189',
        'email': 'ir@goldengateholdings.com',
        'industry': 'Investment Management',
        'annual_revenue_band': '$1M-$5M',
        'employee_count': 12,
        'primary_contact': 'James Chen, Managing Director',
        'tax_id': 'XX-XXX4521',
        'risk_score': 62,
        'kyc_status': 'current',
        'last_review_date': '2023-11-20',
        'account_opened': '2019-04-01',
        'account_balance': 1245000.00,
    },
    'ACC-010': {
        'id': 'ACC-010',
        'name': 'Harbor Real Estate',
        'type': 'business',
        'status': 'active',
        'registration_number': 'LLC-2020-62341',
        'jurisdiction': 'California',
        'incorporation_date': '2020-02-28',
        'registered_agent': 'Pacific Registered Agents',
        'address': '101 California Street, Suite 3800, San Francisco, CA 94111',
        'phone': '+1-415-555-0231',
        'email': 'closings@harborrealestate.com',
        'industry': 'Real Estate',
        'annual_revenue_band': '$5M-$10M',
        'employee_count': 22,
        'primary_contact': 'Linda Wu, Broker',
        'tax_id': 'XX-XXX3392',
        'risk_score': 25,
        'kyc_status': 'current',
        'last_review_date': '2023-06-10',
        'account_opened': '2020-03-15',
        'account_balance': 892000.00,
    },
    'ACC-011': {
        'id': 'ACC-011',
        'name': 'Cafe Milano',
        'type': 'business',
        'status': 'active',
        'registration_number': 'DBA-2021-08823',
        'jurisdiction': 'California',
        'incorporation_date': '2021-04-12',
        'registered_agent': 'Self',
        'address': '2154 Union Street, San Francisco, CA 94123',
        'phone': '+1-415-555-0198',
        'email': 'owner@cafemilano-sf.com',
        'industry': 'Food & Beverage',
        'annual_revenue_band': '$500K-$1M',
        'employee_count': 9,
        'primary_contact': 'Giuseppe Rossi, Owner',
        'tax_id': 'XX-XXX1147',
        'risk_score': 15,
        'kyc_status': 'current',
        'last_review_date': '2023-05-20',
        'account_opened': '2021-05-01',
        'account_balance': 34500.00,
    },
    'ACC-012': {
        'id': 'ACC-012',
        'name': 'Island Getaway Travel',
        'type': 'business',
        'status': 'active',
        'registration_number': 'LLC-2020-71456',
        'jurisdiction': 'Florida',
        'incorporation_date': '2020-08-19',
        'registered_agent': 'Sunshine State Filings',
        'address': '800 Brickell Ave, Suite 1100, Miami, FL 33131',
        'phone': '+1-305-555-0156',
        'email': 'bookings@islandgetaway.travel',
        'industry': 'Travel & Tourism',
        'annual_revenue_band': '$1M-$5M',
        'employee_count': 14,
        'primary_contact': 'Carmen Reyes, General Manager',
        'tax_id': 'XX-XXX8834',
        'risk_score': 20,
        'kyc_status': 'current',
        'last_review_date': '2023-09-05',
        'account_opened': '2020-09-01',
        'account_balance': 67800.00,
    },
    'ACC-013': {
        'id': 'ACC-013',
        'name': 'City Newsstand',
        'type': 'business',
        'status': 'active',
        'registration_number': 'DBA-2015-04412',
        'jurisdiction': 'New York',
        'incorporation_date': '2015-06-01',
        'registered_agent': 'Self',
        'address': '350 Fifth Avenue, Lobby Level, New York, NY 10118',
        'phone': '+1-212-555-0133',
        'email': 'citynewsstand@aol.com',
        'industry': 'Retail',
        'annual_revenue_band': '$100K-$500K',
        'employee_count': 3,
        'primary_contact': 'Frank Abagnale Jr, Owner',
        'tax_id': 'XX-XXX2201',
        'risk_score': 10,
        'kyc_status': 'current',
        'last_review_date': '2022-12-01',
        'account_opened': '2015-07-01',
        'account_balance': 8900.00,
    },
    'ACC-014': {
        'id': 'ACC-014',
        'name': 'Metro Dry Cleaning',
        'type': 'business',
        'status': 'active',
        'registration_number': 'DBA-2018-12290',
        'jurisdiction': 'New York',
        'incorporation_date': '2018-03-22',
        'registered_agent': 'Self',
        'address': '891 Lexington Ave, New York, NY 10065',
        'phone': '+1-212-555-0177',
        'email': 'metrodrycleaning@gmail.com',
        'industry': 'Laundry Services',
        'annual_revenue_band': '$100K-$500K',
        'employee_count': 5,
        'primary_contact': 'Tony Park, Owner',
        'tax_id': 'XX-XXX6678',
        'risk_score': 12,
        'kyc_status': 'current',
        'last_review_date': '2023-02-15',
        'account_opened': '2018-04-01',
        'account_balance': 15200.00,
    },
    'ACC-015': {
        'id': 'ACC-015',
        'name': 'Pine Street Deli',
        'type': 'business',
        'status': 'active',
        'registration_number': 'DBA-2019-09917',
        'jurisdiction': 'Illinois',
        'incorporation_date': '2019-07-14',
        'registered_agent': 'Self',
        'address': '47 W Pine Street, Chicago, IL 60610',
        'phone': '+1-312-555-0129',
        'email': 'pinestreetdeli@outlook.com',
        'industry': 'Food & Beverage',
        'annual_revenue_band': '$100K-$500K',
        'employee_count': 4,
        'primary_contact': 'Sam Goldstein, Owner',
        'tax_id': 'XX-XXX3345',
        'risk_score': 8,
        'kyc_status': 'current',
        'last_review_date': '2023-04-10',
        'account_opened': '2019-08-01',
        'account_balance': 11300.00,
    },
}

# =============================================================================
# Exchange Rates
# =============================================================================

# Simplified: one rate per currency pair (mid-January 2024 approximations).
_FX_RATES: dict[tuple[str, str], float] = {
    ('EUR', 'USD'): 1.0847,
    ('GBP', 'USD'): 1.2693,
    ('CHF', 'USD'): 1.1782,
    ('USD', 'EUR'): 0.9219,
    ('USD', 'GBP'): 0.7878,
    ('USD', 'CHF'): 0.8488,
}


def _to_usd(amount: float, currency: str) -> float:
    """Convert an amount to USD."""
    if currency == 'USD':
        return amount
    rate = _FX_RATES[(currency, 'USD')]
    return round(amount * rate, 2)


# =============================================================================
# Transaction Data
# =============================================================================

# Compact definitions: (txn_id, from, to, amount, currency, date, memo, type)
# Expanded into verbose records by _build_transaction().
# Batch transactions use type='batch_wire' and have a batch_id.
_COMPACT_TXNS: list[tuple[str, str, str, float, str, str, str, str]] = [
    # --- Layer 0: ACC-001 outgoing ---
    (
        'TXN-001',
        'ACC-001',
        'ACC-002',
        43580.00,
        'EUR',
        '2024-01-15',
        'Consulting services - Q1 retainer',
        'wire',
    ),
    (
        'TXN-002',
        'ACC-001',
        'ACC-003',
        25050.00,
        'GBP',
        '2024-01-16',
        'Shipping contract - Baltic route',
        'wire',
    ),
    (
        'TXN-003',
        'ACC-001',
        'ACC-004',
        23150.00,
        'USD',
        '2024-01-17',
        'Personal transfer - living expenses',
        'wire',
    ),
    # --- Layer 0: ACC-001 small outgoing (noise, all below threshold) ---
    (
        'TXN-034',
        'ACC-001',
        'ACC-011',
        1200.00,
        'USD',
        '2024-01-15',
        'Catering deposit - private event',
        'ach',
    ),
    (
        'TXN-035',
        'ACC-001',
        'ACC-013',
        850.00,
        'USD',
        '2024-01-16',
        'Newspaper subscription - annual',
        'ach',
    ),
    (
        'TXN-036',
        'ACC-001',
        'ACC-014',
        2100.00,
        'USD',
        '2024-01-18',
        'Suit dry cleaning - monthly',
        'ach',
    ),
    (
        'TXN-037',
        'ACC-001',
        'ACC-015',
        400.00,
        'USD',
        '2024-01-19',
        'Lunch - client meeting',
        'ach',
    ),
    # --- Layer 1: ACC-002 outgoing ---
    # TXN-004 is a BATCH wire — real recipients hidden behind BATCH-001
    (
        'TXN-004',
        'ACC-002',
        'BATCH',
        28400.00,
        'USD',
        '2024-01-19',
        'Batch wire - multiple payees',
        'batch_wire',
    ),
    (
        'TXN-005',
        'ACC-002',
        'ACC-006',
        14570.00,
        'EUR',
        '2024-01-20',
        'Marketing campaign - Q1 digital',
        'wire',
    ),
    (
        'TXN-006',
        'ACC-002',
        'ACC-013',
        2100.00,
        'USD',
        '2024-01-19',
        'Office supplies and newspapers',
        'ach',
    ),
    (
        'TXN-038',
        'ACC-002',
        'ACC-014',
        3200.00,
        'USD',
        '2024-01-20',
        'Uniform cleaning - warehouse staff',
        'ach',
    ),
    # --- Layer 1: ACC-003 outgoing ---
    (
        'TXN-007',
        'ACC-003',
        'ACC-005',
        18500.00,
        'USD',
        '2024-01-20',
        'Freight forwarding - container lot',
        'wire',
    ),
    (
        'TXN-008',
        'ACC-003',
        'ACC-007',
        9825.00,
        'GBP',
        '2024-01-21',
        'Warehouse rental - Q1 lease',
        'wire',
    ),
    (
        'TXN-009',
        'ACC-003',
        'ACC-014',
        1950.00,
        'USD',
        '2024-01-20',
        'Uniform cleaning service',
        'ach',
    ),
    (
        'TXN-039',
        'ACC-003',
        'ACC-015',
        800.00,
        'USD',
        '2024-01-21',
        'Staff lunch order',
        'ach',
    ),
    # --- Layer 1: ACC-004 outgoing ---
    # TXN-010 is a BATCH wire — real recipients hidden behind BATCH-002
    (
        'TXN-010',
        'ACC-004',
        'BATCH',
        13600.00,
        'USD',
        '2024-01-22',
        'Batch wire - media + investment',
        'batch_wire',
    ),
    (
        'TXN-011',
        'ACC-004',
        'ACC-008',
        9750.00,
        'USD',
        '2024-01-21',
        'Currency exchange - BRL to USD',
        'wire',
    ),
    (
        'TXN-040',
        'ACC-004',
        'ACC-011',
        1200.00,
        'USD',
        '2024-01-22',
        'Event catering deposit',
        'ach',
    ),
    (
        'TXN-041',
        'ACC-004',
        'ACC-012',
        3400.00,
        'USD',
        '2024-01-23',
        'Travel booking - Cancun package',
        'ach',
    ),
    # --- Layer 2: ACC-005 outgoing ---
    (
        'TXN-012',
        'ACC-005',
        'ACC-009',
        38495.00,
        'EUR',
        '2024-01-24',
        'Investment deposit - Q1 2024 tranche',
        'wire',
    ),
    (
        'TXN-013',
        'ACC-005',
        'ACC-010',
        6200.00,
        'USD',
        '2024-01-25',
        'Property deposit - Unit 4B',
        'wire',
    ),
    (
        'TXN-042',
        'ACC-005',
        'ACC-015',
        900.00,
        'USD',
        '2024-01-24',
        'Staff lunch catering',
        'ach',
    ),
    # --- Layer 2: ACC-006 outgoing ---
    (
        'TXN-014',
        'ACC-006',
        'ACC-009',
        22300.00,
        'USD',
        '2024-01-25',
        'Equity purchase - Series B',
        'wire',
    ),
    (
        'TXN-015',
        'ACC-006',
        'ACC-011',
        2800.00,
        'EUR',
        '2024-01-26',
        'Catering contract - annual retainer',
        'ach',
    ),
    # --- Layer 2: ACC-007 outgoing ---
    (
        'TXN-016',
        'ACC-007',
        'ACC-009',
        10250.00,
        'USD',
        '2024-01-26',
        'Fleet financing - vehicle lease',
        'wire',
    ),
    (
        'TXN-017',
        'ACC-007',
        'ACC-015',
        1800.00,
        'USD',
        '2024-01-25',
        'Lunch catering - staff event',
        'ach',
    ),
    # --- Layer 2: ACC-008 outgoing ---
    (
        'TXN-018',
        'ACC-008',
        'ACC-009',
        7150.00,
        'CHF',
        '2024-01-27',
        'Wire transfer - client remittance',
        'wire',
    ),
    (
        'TXN-019',
        'ACC-008',
        'ACC-012',
        1500.00,
        'USD',
        '2024-01-26',
        'Travel booking - Caribbean package',
        'ach',
    ),
    # --- Noise: incoming from external accounts ---
    (
        'TXN-020',
        'EXT-101',
        'ACC-002',
        85000.00,
        'USD',
        '2024-01-10',
        'Import duties settlement - FY2023',
        'wire',
    ),
    (
        'TXN-021',
        'EXT-102',
        'ACC-003',
        62400.00,
        'EUR',
        '2024-01-08',
        'Charter party payment - MV Nordic Star',
        'wire',
    ),
    (
        'TXN-022',
        'EXT-103',
        'ACC-005',
        34750.00,
        'USD',
        '2024-01-12',
        'Strategy engagement - Phase 2 milestone',
        'wire',
    ),
    (
        'TXN-023',
        'EXT-104',
        'ACC-006',
        19200.00,
        'USD',
        '2024-01-11',
        'Ad campaign management - Nov/Dec',
        'ach',
    ),
    (
        'TXN-024',
        'EXT-105',
        'ACC-007',
        27800.00,
        'USD',
        '2024-01-09',
        'Warehousing contract - quarterly',
        'wire',
    ),
    (
        'TXN-025',
        'EXT-106',
        'ACC-008',
        15600.00,
        'CHF',
        '2024-01-13',
        'FX settlement - batch 2024-01',
        'wire',
    ),
    (
        'TXN-026',
        'EXT-107',
        'ACC-009',
        120000.00,
        'USD',
        '2024-01-05',
        'Capital call - Fund III',
        'wire',
    ),
    (
        'TXN-027',
        'EXT-108',
        'ACC-004',
        41300.00,
        'USD',
        '2024-01-14',
        'Trust distribution - Q4 2023',
        'wire',
    ),
    (
        'TXN-028',
        'EXT-109',
        'ACC-001',
        155000.00,
        'EUR',
        '2024-01-03',
        'Trade consortium dividend',
        'wire',
    ),
    (
        'TXN-029',
        'EXT-110',
        'ACC-005',
        22000.00,
        'USD',
        '2024-01-07',
        'Advisory retainer - Jan 2024',
        'ach',
    ),
    (
        'TXN-030',
        'EXT-111',
        'ACC-009',
        55000.00,
        'USD',
        '2024-01-06',
        'LP commitment - tranche 2',
        'wire',
    ),
    (
        'TXN-031',
        'EXT-112',
        'ACC-009',
        38500.00,
        'USD',
        '2024-01-11',
        'Co-investment - Project Evergreen',
        'wire',
    ),
    (
        'TXN-032',
        'EXT-113',
        'ACC-006',
        11750.00,
        'USD',
        '2024-01-09',
        'Creative services - website redesign',
        'ach',
    ),
    (
        'TXN-033',
        'EXT-114',
        'ACC-002',
        29300.00,
        'GBP',
        '2024-01-12',
        'Commodity futures settlement',
        'wire',
    ),
    # --- More noise from externals (to fill pages) ---
    (
        'TXN-043',
        'EXT-115',
        'ACC-001',
        8500.00,
        'USD',
        '2024-01-04',
        'Energy consulting retainer',
        'wire',
    ),
    (
        'TXN-044',
        'EXT-116',
        'ACC-001',
        12000.00,
        'EUR',
        '2024-01-06',
        'Textile import commission',
        'wire',
    ),
    (
        'TXN-045',
        'EXT-117',
        'ACC-001',
        3200.00,
        'EUR',
        '2024-01-09',
        'Freight brokerage fee',
        'ach',
    ),
    (
        'TXN-046',
        'EXT-118',
        'ACC-002',
        45000.00,
        'USD',
        '2024-01-07',
        'Coral Bay export contract - Phase 1',
        'wire',
    ),
    (
        'TXN-047',
        'EXT-119',
        'ACC-002',
        18700.00,
        'USD',
        '2024-01-14',
        'Advisory fee - M&A due diligence',
        'wire',
    ),
    (
        'TXN-048',
        'EXT-120',
        'ACC-003',
        22500.00,
        'EUR',
        '2024-01-05',
        'Mineral transport contract',
        'wire',
    ),
    (
        'TXN-049',
        'EXT-121',
        'ACC-003',
        9800.00,
        'GBP',
        '2024-01-11',
        'Cargo handling services - Q4',
        'wire',
    ),
    (
        'TXN-050',
        'EXT-122',
        'ACC-005',
        15600.00,
        'USD',
        '2024-01-09',
        'Strategy workshop facilitation',
        'wire',
    ),
    (
        'TXN-051',
        'EXT-123',
        'ACC-005',
        7200.00,
        'USD',
        '2024-01-13',
        'Holding company admin fee',
        'ach',
    ),
    (
        'TXN-052',
        'EXT-115',
        'ACC-009',
        28000.00,
        'USD',
        '2024-01-08',
        'Energy sector investment',
        'wire',
    ),
    (
        'TXN-053',
        'EXT-119',
        'ACC-009',
        19500.00,
        'USD',
        '2024-01-13',
        'Advisory placement fee',
        'wire',
    ),
]

# =============================================================================
# Batch Data
# =============================================================================

# Batch transactions hide real recipients — the parent transaction only shows
# a total amount. You must call get_batch_details() to see where the money went.
_BATCHES: dict[str, dict[str, Any]] = {
    'BATCH-001': {
        'batch_id': 'BATCH-001',
        'parent_transaction': 'TXN-004',
        'from_account': 'ACC-002',
        'from_account_name': 'Oceanic Trading LLC',
        'total_amount': 28400.00,
        'currency': 'USD',
        'date': '2024-01-19',
        'status': 'completed',
        'sub_transactions': [
            {
                'id': 'TXN-004-A',
                'to_account': 'ACC-005',
                'to_account_name': 'Sunrise Consulting',
                'amount': 22400.00,
                'currency': 'USD',
                'memo': 'Subcontractor payment - Project Alpha',
                'status': 'completed',
            },
            {
                'id': 'TXN-004-B',
                'to_account': 'ACC-010',
                'to_account_name': 'Harbor Real Estate',
                'amount': 6000.00,
                'currency': 'USD',
                'memo': 'Property escrow deposit - Lot 7',
                'status': 'completed',
            },
        ],
    },
    'BATCH-002': {
        'batch_id': 'BATCH-002',
        'parent_transaction': 'TXN-010',
        'from_account': 'ACC-004',
        'from_account_name': 'Maria Santos',
        'total_amount': 13600.00,
        'currency': 'USD',
        'date': '2024-01-22',
        'status': 'completed',
        'sub_transactions': [
            {
                'id': 'TXN-010-A',
                'to_account': 'ACC-006',
                'to_account_name': 'Digital Media Partners',
                'amount': 8600.00,
                'currency': 'USD',
                'memo': 'Media production - promotional video',
                'status': 'completed',
            },
            {
                'id': 'TXN-010-B',
                'to_account': 'ACC-009',
                'to_account_name': 'Golden Gate Holdings',
                'amount': 5000.00,
                'currency': 'USD',
                'memo': 'Investment contribution - Series A',
                'status': 'completed',
            },
        ],
    },
}

_TXN_TO_BATCH: dict[str, str] = {
    'TXN-004': 'BATCH-001',
    'TXN-010': 'BATCH-002',
}

_SWIFT_CODES = ('CHASUS33', 'BOFAUS3N', 'CITIUS33', 'WFBIUS6S')
_OFFICER_IDS = ('OFC-0892', 'OFC-1247', 'OFC-0651', 'OFC-0433')


def _build_transaction(
    idx: int,
    txn_id: str,
    from_acc: str,
    to_acc: str,
    amount: float,
    currency: str,
    date: str,
    memo: str,
    txn_type: str,
) -> dict[str, Any]:
    """Expand compact transaction tuple into a verbose API-style record."""
    day = int(date.split('-')[2])
    settlement_day = min(day + 2, 28)
    prefix = {
        'wire': 'WR',
        'ach': 'ACH',
        'check': 'CHK',
        'internal': 'INT',
        'batch_wire': 'BW',
    }

    record: dict[str, Any] = {
        'id': txn_id,
        'reference': f'{prefix[txn_type]}-2024-{1000 + idx:04d}-{from_acc}-{to_acc}',
        'from_account': from_acc,
        'from_account_name': _ALL_NAMES.get(from_acc, from_acc),
        'amount': amount,
        'currency': currency,
        'date': date,
        'settlement_date': f'{date[:8]}{settlement_day:02d}',
        'type': txn_type,
        'status': 'completed',
        'intermediary_bank': f'SWIFT: {_SWIFT_CODES[idx % len(_SWIFT_CODES)]}',
        'memo': memo,
        'compliance_flags': [],
        'processing_fee': round(amount * 0.0015, 2),
        'batch_id': f'BT-{date.replace("-", "")}-{(idx % 5) + 1}',
        'officer_id': _OFFICER_IDS[idx % len(_OFFICER_IDS)],
    }

    if txn_type == 'batch_wire':
        # Batch wires show "Multiple payees" — real recipients are in the batch.
        batch_id = _TXN_TO_BATCH[txn_id]
        record['to_account'] = 'MULTIPLE'
        record['to_account_name'] = 'Multiple payees (see batch details)'
        record['batch_id'] = batch_id
    else:
        record['to_account'] = to_acc
        record['to_account_name'] = _ALL_NAMES.get(to_acc, to_acc)

    return record


_TRANSACTIONS: dict[str, dict[str, Any]] = {
    t[0]: _build_transaction(i, *t) for i, t in enumerate(_COMPACT_TXNS)
}

_flagged_accounts: list[dict[str, Any]] = []


# =============================================================================
# Ground Truth (computed from the transaction data)
# =============================================================================


def _compute_ground_truth() -> list[tuple[str, str, int, float]]:
    """BFS from ACC-001 up to 3 hops, return convergence points.

    Only follows outgoing transactions >= AMOUNT_THRESHOLD (in USD).
    Expands batch transactions to find real recipients.
    Returns (account_id, name, source_count, total_inflow_usd) sorted by inflow desc.
    """
    inflow: dict[str, float] = {}
    sources: dict[str, set[str]] = {}

    def _record_edge(
        from_acct: str, to_acct: str, amount: float, currency: str
    ) -> list[str]:
        """Record an edge if above threshold. Return list of new destinations."""
        usd = _to_usd(amount, currency)
        if usd < AMOUNT_THRESHOLD:
            return []
        inflow[to_acct] = inflow.get(to_acct, 0) + usd
        if to_acct not in sources:
            sources[to_acct] = set()
        sources[to_acct].add(from_acct)
        return [to_acct]

    visited: set[str] = set()
    current_layer = ['ACC-001']

    for _ in range(3):
        next_layer: list[str] = []
        for acct in current_layer:
            if acct in visited:
                continue
            visited.add(acct)
            for txn in _TRANSACTIONS.values():
                if txn['from_account'] != acct:
                    continue
                if txn['type'] == 'batch_wire':
                    # Expand batch to find real recipients
                    batch_id = txn['batch_id']
                    batch = _BATCHES[batch_id]
                    for sub in batch['sub_transactions']:
                        dests = _record_edge(
                            acct, sub['to_account'], sub['amount'], sub['currency']
                        )
                        for d in dests:
                            if d not in visited:
                                next_layer.append(d)
                else:
                    dests = _record_edge(
                        acct, txn['to_account'], txn['amount'], txn['currency']
                    )
                    for d in dests:
                        if d not in visited:
                            next_layer.append(d)
        current_layer = next_layer

    convergence: list[tuple[str, str, int, float]] = []
    for acct_id, srcs in sources.items():
        if len(srcs) >= 2:
            name = _ACCOUNT_NAMES.get(acct_id, acct_id)
            convergence.append((acct_id, name, len(srcs), round(inflow[acct_id], 2)))

    convergence.sort(key=lambda x: (-x[3], x[0]))
    return convergence


EXPECTED_CONVERGENCE = _compute_ground_truth()
EXPECTED_FLAG_IDS = {c[0] for c in EXPECTED_CONVERGENCE}


# =============================================================================
# Mock Tools
# =============================================================================


def list_account_transactions(account_id: str, page: int = 1) -> dict[str, Any]:
    """List transactions for an account, paginated (3 per page).

    Returns transaction summaries — use get_transaction() for full details
    including amounts and currencies.

    Args:
        account_id: The account ID to look up (e.g. "ACC-001").
        page: Page number (1-indexed). Each page returns up to 3 transactions.

    Returns:
        A dict with:
        - transactions: list of transaction summaries (id, direction, date, counterparty, category, status)
        - page: current page number
        - total_pages: total number of pages
        - has_more: whether there are more pages
    """
    all_results: list[dict[str, Any]] = []
    for txn_id, txn in _TRANSACTIONS.items():
        if txn['from_account'] == account_id:
            counterparty = txn['to_account_name']
            if txn['type'] == 'batch_wire':
                counterparty = 'Multiple payees (batch wire)'
            all_results.append(
                {
                    'transaction_id': txn_id,
                    'direction': 'outgoing',
                    'date': txn['date'],
                    'counterparty': counterparty,
                    'category': f'{txn["type"]}_transfer',
                    'status': txn['status'],
                }
            )
        elif txn['to_account'] == account_id:
            all_results.append(
                {
                    'transaction_id': txn_id,
                    'direction': 'incoming',
                    'date': txn['date'],
                    'counterparty': txn['from_account_name'],
                    'category': f'{txn["type"]}_transfer',
                    'status': txn['status'],
                }
            )
    all_results.sort(key=lambda x: x['date'])

    total = len(all_results)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_results = all_results[start:end]

    return {
        'transactions': page_results,
        'page': page,
        'total_pages': total_pages,
        'has_more': page < total_pages,
    }


def get_transaction(transaction_id: str) -> dict[str, Any]:
    """Get full details of a specific transaction.

    Note: batch wire transfers show "Multiple payees" as the recipient.
    Use get_batch_details() with the batch_id to see individual recipients.

    Args:
        transaction_id: The transaction ID (e.g. "TXN-001").

    Returns:
        Full transaction record including amount, currency, accounts, dates, and metadata.
        Batch wires include a batch_id field — call get_batch_details() to expand.
    """
    txn = _TRANSACTIONS.get(transaction_id)
    if txn is None:
        return {'error': f'Transaction {transaction_id} not found'}
    return txn


def get_account_info(account_id: str) -> dict[str, Any]:
    """Get account details.

    Args:
        account_id: The account ID to look up (e.g. "ACC-001").

    Returns:
        Full account record including name, type, address, risk score, and more.
    """
    account = _ACCOUNTS.get(account_id)
    if account is None:
        return {'error': f'Account {account_id} not found'}
    return account


def get_exchange_rate(from_currency: str, to_currency: str) -> dict[str, Any]:
    """Get the exchange rate between two currencies.

    Args:
        from_currency: Source currency code (e.g. "EUR").
        to_currency: Target currency code (e.g. "USD").

    Returns:
        A dict with from_currency, to_currency, and rate.
    """
    if from_currency == to_currency:
        return {'from_currency': from_currency, 'to_currency': to_currency, 'rate': 1.0}
    rate = _FX_RATES.get((from_currency, to_currency))
    if rate is None:
        return {'error': f'No rate available for {from_currency}/{to_currency}'}
    return {'from_currency': from_currency, 'to_currency': to_currency, 'rate': rate}


def get_batch_details(batch_id: str) -> dict[str, Any]:
    """Get the sub-transactions within a batch wire transfer.

    Batch wires bundle multiple payments into a single transaction. This endpoint
    returns the individual recipients and amounts.

    Args:
        batch_id: The batch ID (e.g. "BATCH-001").

    Returns:
        Batch details including sub_transactions with individual recipients and amounts.
    """
    batch = _BATCHES.get(batch_id)
    if batch is None:
        return {'error': f'Batch {batch_id} not found'}
    return batch


def flag_account(account_id: str, reason: str) -> dict[str, Any]:
    """Flag an account for suspicious activity.

    Args:
        account_id: The account ID to flag.
        reason: The reason for flagging.

    Returns:
        Confirmation with the account ID and reason.
    """
    record = {'account_id': account_id, 'reason': reason, 'status': 'flagged'}
    _flagged_accounts.append(record)
    return record


def create_toolset() -> FunctionToolset[None]:
    """Create the transaction investigation toolset."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(list_account_transactions)
    toolset.add_function(get_transaction)
    toolset.add_function(get_account_info)
    toolset.add_function(get_exchange_rate)
    toolset.add_function(get_batch_details)
    toolset.add_function(flag_account)
    return toolset


# =============================================================================
# Agent Factories
# =============================================================================

SYSTEM_PROMPT = (
    'You are a financial crime investigator. Use the available tools to trace '
    'money flows and identify suspicious patterns in transaction networks. '
    'Transaction lists are paginated — make sure to fetch ALL pages. '
    'Some transactions are batch wires — you MUST expand them with get_batch_details '
    'to see where the money actually went. '
    'Foreign currency amounts must be converted to USD using get_exchange_rate '
    'before comparing against thresholds or summing totals.'
)


def create_tool_calling_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    """Create agent with standard tool calling."""
    return Agent(MODEL, toolsets=[toolset], system_prompt=SYSTEM_PROMPT)


def create_code_mode_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    """Create agent with CodeMode (tools as Python functions)."""
    runtime = MontyRuntime()
    code_toolset: CodeModeToolset[None] = CodeModeToolset(
        wrapped=toolset,
        max_retries=MAX_RETRIES,
        runtime=runtime,
    )
    return Agent(MODEL, toolsets=[code_toolset], system_prompt=SYSTEM_PROMPT)


# =============================================================================
# Metrics Collection
# =============================================================================


@dataclass
class RunMetrics:
    """Metrics collected from an agent run."""

    mode: str
    request_count: int
    input_tokens: int
    output_tokens: int
    retry_count: int
    output: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def extract_metrics(result: AgentRunResult[str], mode: str) -> RunMetrics:
    """Extract metrics from agent result."""
    request_count = 0
    input_tokens = 0
    output_tokens = 0
    retry_count = 0

    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            request_count += 1
            if msg.usage:
                input_tokens += msg.usage.input_tokens or 0
                output_tokens += msg.usage.output_tokens or 0
        for part in getattr(msg, 'parts', []):
            if isinstance(part, RetryPromptPart):
                retry_count += 1

    return RunMetrics(
        mode=mode,
        request_count=request_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        retry_count=retry_count,
        output=result.output,
    )


# =============================================================================
# Run Functions
# =============================================================================


async def run_tool_calling(toolset: FunctionToolset[None]) -> RunMetrics:
    """Run with standard tool calling."""
    global _flagged_accounts
    _flagged_accounts = []

    with logfire.span('tool_calling'):
        agent = create_tool_calling_agent(toolset)
        result = await agent.run(PROMPT)
    return extract_metrics(result, 'tool_calling')


async def run_code_mode(toolset: FunctionToolset[None]) -> RunMetrics:
    """Run with CodeMode tool calling."""
    global _flagged_accounts
    _flagged_accounts = []

    with logfire.span('code_mode_tool_calling'):
        agent = create_code_mode_agent(toolset)
        code_toolset = agent.toolsets[0]
        async with code_toolset:
            result = await agent.run(PROMPT)
    return extract_metrics(result, 'code_mode')


# =============================================================================
# Main Demo
# =============================================================================


def verify_results(mode: str) -> None:
    """Check flagged accounts against ground truth and log to logfire."""
    flagged_ids = {r['account_id'] for r in _flagged_accounts}
    correct: set[str] = set()
    missed: set[str] = set()
    spurious: set[str] = set()
    for fid in flagged_ids:
        if fid in EXPECTED_FLAG_IDS:
            correct.add(fid)
        else:
            spurious.add(fid)
    for eid in EXPECTED_FLAG_IDS:
        if eid not in flagged_ids:
            missed.add(eid)

    if flagged_ids == EXPECTED_FLAG_IDS:
        logfire.info(
            '{mode} verification: PASS — correctly flagged {count} accounts: {ids}',
            mode=mode,
            count=len(correct),
            ids=', '.join(sorted(correct)),
        )
    else:
        logfire.error(
            '{mode} verification: FAIL — correct: {correct}, missed: {missed}, spurious: {spurious}',
            mode=mode,
            correct=', '.join(sorted(correct)) or 'none',
            missed=', '.join(sorted(missed)) or 'none',
            spurious=', '.join(sorted(spurious)) or 'none',
        )


def log_metrics(metrics: RunMetrics) -> None:
    """Log metrics to logfire."""
    logfire.info(
        '{mode} completed: {requests} requests, {tokens} tokens',
        mode=metrics.mode,
        requests=metrics.request_count,
        tokens=metrics.total_tokens,
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        retries=metrics.retry_count,
    )


async def main() -> None:
    logfire.configure(service_name='code-mode-follow-the-money')
    logfire.instrument_pydantic_ai()

    toolset = create_toolset()

    # Print ground truth for debugging
    print('Expected convergence points:')
    for acct_id, name, src_count, total in EXPECTED_CONVERGENCE:
        print(f'  {acct_id} ({name}): {src_count} sources, ${total:,.2f} USD inflow')
    print()

    # with logfire.span('demo_tool_calling'):
    #     trad = await run_tool_calling(toolset)
    # log_metrics(trad)
    # verify_results('tool_calling')

    with logfire.span('demo_code_mode'):
        code = await run_code_mode(toolset)
    log_metrics(code)
    verify_results('code_mode')

    print('View traces: https://logfire.pydantic.dev')


if __name__ == '__main__':
    asyncio.run(main())
