"""Nory x402 Payment Tools for Pydantic AI.

Tools for AI agents to make payments using the x402 HTTP protocol.
Supports Solana and 7 EVM chains with sub-400ms settlement.
"""

from dataclasses import KW_ONLY, dataclass
from typing import Literal

import httpx
from pydantic import TypeAdapter
from typing_extensions import Any, TypedDict

from pydantic_ai.tools import Tool

__all__ = (
    'nory_get_payment_requirements_tool',
    'nory_verify_payment_tool',
    'nory_settle_payment_tool',
    'nory_lookup_transaction_tool',
    'nory_health_check_tool',
    'nory_x402_tools',
)

NORY_API_BASE = 'https://noryx402.com'

NoryNetwork = Literal[
    'solana-mainnet',
    'solana-devnet',
    'base-mainnet',
    'polygon-mainnet',
    'arbitrum-mainnet',
    'optimism-mainnet',
    'avalanche-mainnet',
    'sei-mainnet',
    'iotex-mainnet',
]


class PaymentRequirements(TypedDict, total=False):
    """Payment requirements returned by Nory."""

    amount: str
    """The amount to pay."""
    currency: str
    """The currency (e.g., USDC)."""
    networks: list[str]
    """Supported networks for payment."""
    wallet_address: str
    """The wallet address to send payment to."""


class VerificationResult(TypedDict, total=False):
    """Payment verification result."""

    valid: bool
    """Whether the payment is valid."""
    payer: str
    """The payer's address."""
    amount: str
    """The payment amount."""


class SettlementResult(TypedDict, total=False):
    """Payment settlement result."""

    success: bool
    """Whether settlement succeeded."""
    transaction_id: str
    """The transaction ID/signature."""
    network: str
    """The network used."""


class TransactionStatus(TypedDict, total=False):
    """Transaction status."""

    status: str
    """Transaction status (pending, confirmed, failed)."""
    confirmations: int
    """Number of confirmations."""
    transaction_id: str
    """The transaction ID."""


class HealthStatus(TypedDict, total=False):
    """Nory service health status."""

    status: str
    """Service status."""
    networks: list[str]
    """Supported networks."""


payment_requirements_ta = TypeAdapter(PaymentRequirements)
verification_result_ta = TypeAdapter(VerificationResult)
settlement_result_ta = TypeAdapter(SettlementResult)
transaction_status_ta = TypeAdapter(TransactionStatus)
health_status_ta = TypeAdapter(HealthStatus)


@dataclass
class NoryGetPaymentRequirementsTool:
    """Tool to get x402 payment requirements for a resource."""

    _: KW_ONLY
    api_key: str | None = None
    """Nory API key (optional for public endpoints)."""

    async def __call__(
        self,
        resource: str,
        amount: str,
        network: NoryNetwork | None = None,
    ) -> PaymentRequirements:
        """Get x402 payment requirements for accessing a paid resource.

        Use this when you encounter an HTTP 402 Payment Required response
        and need to know how much to pay and where to send payment.

        Args:
            resource: The resource path requiring payment (e.g., /api/premium/data).
            amount: Amount in human-readable format (e.g., '0.10' for $0.10 USDC).
            network: Preferred blockchain network.

        Returns:
            Payment requirements including amount, supported networks, and wallet address.
        """
        params: dict[str, str] = {'resource': resource, 'amount': amount}
        if network:
            params['network'] = network

        headers: dict[str, str] = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'{NORY_API_BASE}/api/x402/requirements',
                params=params,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return payment_requirements_ta.validate_python(response.json())


@dataclass
class NoryVerifyPaymentTool:
    """Tool to verify a signed payment transaction."""

    _: KW_ONLY
    api_key: str | None = None
    """Nory API key (optional for public endpoints)."""

    async def __call__(self, payload: str) -> VerificationResult:
        """Verify a signed payment transaction before settlement.

        Use this to validate that a payment transaction is correct
        before submitting it to the blockchain.

        Args:
            payload: Base64-encoded payment payload containing signed transaction.

        Returns:
            Verification result including validity and payer info.
        """
        headers: dict[str, str] = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{NORY_API_BASE}/api/x402/verify',
                json={'payload': payload},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return verification_result_ta.validate_python(response.json())


@dataclass
class NorySettlePaymentTool:
    """Tool to settle a payment on-chain."""

    _: KW_ONLY
    api_key: str | None = None
    """Nory API key (optional for public endpoints)."""

    async def __call__(self, payload: str) -> SettlementResult:
        """Settle a payment on-chain.

        Use this to submit a verified payment transaction to the blockchain.
        Settlement typically completes in under 400ms.

        Args:
            payload: Base64-encoded payment payload.

        Returns:
            Settlement result including transaction ID.
        """
        headers: dict[str, str] = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{NORY_API_BASE}/api/x402/settle',
                json={'payload': payload},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return settlement_result_ta.validate_python(response.json())


@dataclass
class NoryLookupTransactionTool:
    """Tool to look up transaction status."""

    _: KW_ONLY
    api_key: str | None = None
    """Nory API key (optional for public endpoints)."""

    async def __call__(self, transaction_id: str, network: NoryNetwork) -> TransactionStatus:
        """Look up transaction status.

        Use this to check the status of a previously submitted payment.

        Args:
            transaction_id: Transaction ID or signature.
            network: Network where the transaction was submitted.

        Returns:
            Transaction details including status and confirmations.
        """
        headers: dict[str, str] = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'{NORY_API_BASE}/api/x402/transactions/{transaction_id}',
                params={'network': network},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return transaction_status_ta.validate_python(response.json())


@dataclass
class NoryHealthCheckTool:
    """Tool to check Nory service health."""

    async def __call__(self) -> HealthStatus:
        """Check Nory service health.

        Use this to verify the payment service is operational
        and see supported networks.

        Returns:
            Health status and supported networks.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(f'{NORY_API_BASE}/api/x402/health', timeout=30)
            response.raise_for_status()
            return health_status_ta.validate_python(response.json())


def nory_get_payment_requirements_tool(api_key: str | None = None) -> Tool[Any]:
    """Creates a tool to get x402 payment requirements.

    Args:
        api_key: Nory API key (optional for public endpoints).
    """
    return Tool[Any](
        NoryGetPaymentRequirementsTool(api_key=api_key).__call__,
        name='nory_get_payment_requirements',
        description='Get x402 payment requirements for accessing a paid resource. Returns amount, supported networks, and wallet address.',
    )


def nory_verify_payment_tool(api_key: str | None = None) -> Tool[Any]:
    """Creates a tool to verify a signed payment transaction.

    Args:
        api_key: Nory API key (optional for public endpoints).
    """
    return Tool[Any](
        NoryVerifyPaymentTool(api_key=api_key).__call__,
        name='nory_verify_payment',
        description='Verify a signed payment transaction before submitting to blockchain.',
    )


def nory_settle_payment_tool(api_key: str | None = None) -> Tool[Any]:
    """Creates a tool to settle a payment on-chain.

    Args:
        api_key: Nory API key (optional for public endpoints).
    """
    return Tool[Any](
        NorySettlePaymentTool(api_key=api_key).__call__,
        name='nory_settle_payment',
        description='Settle a payment on-chain with ~400ms settlement time.',
    )


def nory_lookup_transaction_tool(api_key: str | None = None) -> Tool[Any]:
    """Creates a tool to look up transaction status.

    Args:
        api_key: Nory API key (optional for public endpoints).
    """
    return Tool[Any](
        NoryLookupTransactionTool(api_key=api_key).__call__,
        name='nory_lookup_transaction',
        description='Look up the status of a previously submitted payment transaction.',
    )


def nory_health_check_tool() -> Tool[Any]:
    """Creates a tool to check Nory service health."""
    return Tool[Any](
        NoryHealthCheckTool().__call__,
        name='nory_health_check',
        description='Check Nory service health and see supported networks.',
    )


def nory_x402_tools(api_key: str | None = None) -> list[Tool[Any]]:
    """Creates all Nory x402 payment tools.

    Args:
        api_key: Nory API key (optional for public endpoints).

    Returns:
        A list of all Nory x402 tools.
    """
    return [
        nory_get_payment_requirements_tool(api_key),
        nory_verify_payment_tool(api_key),
        nory_settle_payment_tool(api_key),
        nory_lookup_transaction_tool(api_key),
        nory_health_check_tool(),
    ]
