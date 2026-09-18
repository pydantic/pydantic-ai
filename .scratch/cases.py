"""120 labelled support tickets, for comparing what we put on the wire to Jev.

**How the labels were made, and what they are worth.** They are not anyone's opinion about the ticket
and they were not produced by a model. Each one follows a routing rule written before the cases were,
so a label is a fact about the rule:

    area = billing   the ticket is about money already owed, charged, quoted or refunded
           account   access, membership, permissions or the identity of a user
           bug       the product did something other than what it says it does
           other     none of the above: how-to questions, roadmap, feedback, content errors
    urgent = True    someone is blocked right now, money is moving wrongly right now, or
                     the product is broken for more than the one person writing
             False   everything else, including angry but not blocked

That makes the set worth something for *comparing* wire shapes, which is what it is for: every shape is
scored against the same rule, so a shape that agrees with the rule more often really does read the rule
better. It is worth much less as an absolute accuracy figure, because the rule is ours. Anyone quoting a
number from this outside that comparison should quote the rule with it.

Deliberately included: tickets where the surface words point at the wrong area (a "refund" that is really
a bug report), where anger is not urgency, and where politeness hides a real outage.
"""

from __future__ import annotations

from typing import Literal

Area = Literal['billing', 'account', 'bug', 'other']

# (text, urgent, area)
CASES: list[tuple[str, bool, Area]] = [
    # --- billing, urgent: money moving wrongly right now -------------------------------------------
    ('Second time this month my card was charged twice. Fix it or I cancel.', True, 'billing'),
    ('You billed us for 50 seats, we have 12. Invoice attached.', True, 'billing'),
    ('We were charged the annual plan after cancelling last week.', True, 'billing'),
    ('Our card is being declined on every retry and the account says past due.', True, 'billing'),
    ('I have three duplicate invoices for October, all paid.', True, 'billing'),
    ('The refund you issued came back $40 short.', True, 'billing'),
    ('We are being charged in USD, our contract says EUR.', True, 'billing'),
    ('Invoice 88213 charges for a month we were not customers.', True, 'billing'),
    ('Payment went through twice and both show as pending.', True, 'billing'),
    ('You have taken the deposit but the plan still shows unpaid.', True, 'billing'),
    # --- billing, not urgent -----------------------------------------------------------------------
    ('Can you send a copy of last March invoice for our accountant?', False, 'billing'),
    ('Renewal quote came in 40% higher, we are evaluating alternatives.', False, 'billing'),
    ('How do I add a VAT number to our receipts?', False, 'billing'),
    ('Do you offer annual billing at a discount?', False, 'billing'),
    ('Please change the billing contact to our finance team.', False, 'billing'),
    ('Could we move the invoice date to the first of the month?', False, 'billing'),
    ('Is there a nonprofit rate?', False, 'billing'),
    ('We would like a quote for 200 seats.', False, 'billing'),
    ('Can I get receipts for the whole of last year in one PDF?', False, 'billing'),
    ('What happens to unused credit if we downgrade?', False, 'billing'),
    ('Our PO number needs to appear on invoices from now on.', False, 'billing'),
    ('Do you accept bank transfer instead of card?', False, 'billing'),
    ('I was told there would be a prorated credit, where does it show?', False, 'billing'),
    ('Please cancel the subscription at the end of the term.', False, 'billing'),
    ('Can we split billing across two departments?', False, 'billing'),
    # --- account, urgent: someone is blocked right now ---------------------------------------------
    ('Our whole team is locked out, nobody can sign in since the deploy.', True, 'account'),
    ('Password reset email never arrives, tried three times, blocked all morning.', True, 'account'),
    ('SSO stopped working this morning and no one can get in.', True, 'account'),
    ('My admin removed themselves and now nobody can add users.', True, 'account'),
    ('Two-factor codes are all rejected, I cannot get into my account.', True, 'account'),
    ('The invite link expired and our contractor starts in an hour.', True, 'account'),
    ('Our API keys were rotated without warning and everything is unauthorized.', True, 'account'),
    ('I am locked out after too many attempts and the unlock link 404s.', True, 'account'),
    ('Someone removed my owner role and I cannot restore it.', True, 'account'),
    ('We think an old employee still has access to production.', True, 'account'),
    # --- account, not urgent -----------------------------------------------------------------------
    ('Please add my colleague as a read-only member when you get a chance.', False, 'account'),
    ('How do I transfer ownership of the workspace?', False, 'account'),
    ('Can we enforce two-factor for everyone on the team?', False, 'account'),
    ('I would like to change the email on my account.', False, 'account'),
    ('Is there a way to see who last signed in?', False, 'account'),
    ('Please delete the account of someone who left last month.', False, 'account'),
    ('Can we have more than one owner?', False, 'account'),
    ('How long do sessions last before they expire?', False, 'account'),
    ('We want to rename our workspace.', False, 'account'),
    ('Do you support SCIM provisioning?', False, 'account'),
    ('Can I merge two accounts I created by mistake?', False, 'account'),
    ('What permissions does the member role have exactly?', False, 'account'),
    ('Please remove the pending invite I sent to the wrong address.', False, 'account'),
    ('Is there an audit log I can export?', False, 'account'),
    ('Can I set a default role for new members?', False, 'account'),
    # --- bug, urgent: broken for more than the one person ------------------------------------------
    ('Checkout returns a 500 for every customer right now.', True, 'bug'),
    ('The API has been returning empty arrays for all queries since 09:00.', True, 'bug'),
    ('Webhooks stopped firing entirely about an hour ago.', True, 'bug'),
    ('Every export produces a zero-byte file.', True, 'bug'),
    ('The dashboard shows last week data for everyone on our team.', True, 'bug'),
    ('Search returns no results at all, for any query, for all our users.', True, 'bug'),
    ('Saving a record silently drops half the fields.', True, 'bug'),
    ('The mobile app crashes on launch after the update.', True, 'bug'),
    ('Rate limiting is rejecting requests far below our documented quota.', True, 'bug'),
    ('Scheduled jobs have not run since yesterday evening.', True, 'bug'),
    ('Notifications are going to the wrong customers.', True, 'bug'),
    ('Data from another organisation appeared in our report.', True, 'bug'),
    # --- bug, not urgent ---------------------------------------------------------------------------
    ('The export button is misaligned on Firefox, minor thing.', False, 'bug'),
    ('Tooltip text is cut off on narrow screens.', False, 'bug'),
    ('Sorting by date puts empty values first, which seems wrong.', False, 'bug'),
    ('The dark mode toggle resets when I reload.', False, 'bug'),
    ('Pagination shows page 0 of 3 on the first page.', False, 'bug'),
    ('Timestamps in the CSV are UTC but the UI shows local time.', False, 'bug'),
    ('The search box loses focus after you type the first letter.', False, 'bug'),
    ('Long names overflow their container on the members page.', False, 'bug'),
    ('The keyboard shortcut for save does nothing on Linux.', False, 'bug'),
    ('An archived project still shows in the picker.', False, 'bug'),
    ('The chart legend colours do not match the bars.', False, 'bug'),
    ('Copying a link adds a trailing space.', False, 'bug'),
    ('The unread badge sometimes shows a stale count.', False, 'bug'),
    # --- other, not urgent -------------------------------------------------------------------------
    ('How do I change the timezone shown on my dashboard?', False, 'other'),
    ('Do you have a Terraform provider on the roadmap?', False, 'other'),
    ('Typo on your pricing page: "recieve".', False, 'other'),
    ('Is there an API for the reporting feature?', False, 'other'),
    ('Where can I find the changelog?', False, 'other'),
    ('Do you have a status page I can subscribe to?', False, 'other'),
    ('Your docs example uses an old parameter name.', False, 'other'),
    ('Any plans for a Linux desktop build?', False, 'other'),
    ('Just wanted to say the new editor is great.', False, 'other'),
    ('Is there a community forum?', False, 'other'),
    ('Can you point me at the data retention policy?', False, 'other'),
    ('What regions do you host in?', False, 'other'),
    ('Do you have a partner programme?', False, 'other'),
    ('The onboarding video has no captions.', False, 'other'),
    ('Could you add a keyboard shortcut for the search box?', False, 'other'),
    ('Which browsers do you officially support?', False, 'other'),
    ('Is the SOC 2 report available?', False, 'other'),
    ('Do you publish an OpenAPI spec?', False, 'other'),
    ('Your blog RSS feed seems to be missing recent posts.', False, 'other'),
    ('Can I get a sticker for my laptop?', False, 'other'),
    # --- the ones where the surface words point the wrong way ---------------------------------------
    # Says "refund", is a bug report.
    ('I asked for a refund in the app and it just spins forever.', True, 'bug'),
    # Says "invoice", is a how-to.
    ('Where in the docs is the invoice API documented?', False, 'other'),
    # Says "password", is a feature request.
    ('Would you consider supporting passkeys instead of passwords?', False, 'other'),
    # Angry, but nobody is blocked and nothing is broken.
    ('This is the third time I have had to explain this. Your onboarding is terrible.', False, 'other'),
    # Polite, but the product is down for everyone.
    ('Sorry to bother you, but none of our users can load the site at the moment.', True, 'bug'),
    # Says "charged", is about a quote not a charge.
    ('Before we sign, how much would we be charged for going over the seat limit?', False, 'billing'),
    # Says "locked out", is a how-to about a feature called Lock.
    ('How does the Lock feature stop other people editing a record?', False, 'other'),
    # Says "urgent" but is a scheduling request.
    ('URGENT: please make sure our renewal does not auto-charge next Tuesday.', False, 'billing'),
    # Mentions security, is a docs question.
    ('Does your encryption at rest cover database backups? Asking for our review.', False, 'other'),
    # Reads like a feature request, is an outage.
    ('It would be great if the API returned data again — everything is 503 since noon.', True, 'bug'),
    # Access problem phrased as billing.
    ('We paid for the enterprise plan but the admin settings are still hidden.', True, 'account'),
    # Billing problem phrased as a bug.
    ('There is a bug in your invoice: it charges tax twice on the same line.', True, 'billing'),
    # One person inconvenienced, not an outage.
    ('The app is a bit slow for me this morning, others seem fine.', False, 'bug'),
    # Blocked, but the cause is their own scheduling.
    ('Our trial expired overnight and the team cannot get in.', True, 'account'),
    # Sounds catastrophic, is a question.
    ('What happens to our data if we stop paying? Worried about losing everything.', False, 'other'),
    # --- top-up to 120, same rule ------------------------------------------------------------------
    ('Your system just charged our card for a plan we downgraded from in June.', True, 'billing'),
    ('We have been double-billed on every invoice since the currency change.', True, 'billing'),
    ('A refund for the cancelled add-on has been "processing" for three weeks.', True, 'billing'),
    ('Every login attempt redirects back to the sign-in page for all of us.', True, 'account'),
    ('Our SSO certificate expired and the rotation page will not accept the new one.', True, 'account'),
    ('The owner account was deleted by mistake and now nobody can manage billing.', True, 'account'),
    ('Uploads over 10 MB fail for every user since this morning.', True, 'bug'),
    ('Two-factor prompts are being sent to the wrong phone numbers.', True, 'bug'),
    ('Can you recommend a good book on your query language?', False, 'other'),
    ('Do you offer a student plan?', False, 'billing'),
]

assert len(CASES) == 120, len(CASES)
assert len({text for text, _, _ in CASES}) == 120, 'duplicate ticket text'
