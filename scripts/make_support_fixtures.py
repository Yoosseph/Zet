"""Write tests/fixtures/support/{questions.yaml, emails.csv}: the step 7 fixture set.

LLM-written (Claude), labeled by the same author; for correctness tests and a demo run only, not
for published numbers (see docs/decisions.md, Q12). Rerun to regenerate: python scripts/make_support_fixtures.py
"""
import csv
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "support"

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this email?",
                   "criteria": {"billing": "invoices, payments, charges, refunds",
                                "technical": "bugs, crashes, errors, login problems",
                                "sales": "buying, upgrading, pricing, demos",
                                "other": "anything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this email?",
                "criteria": ["not urgent", "soon", "critical"]},
    "churn_risk": {"type": "score", "instructions": "How likely is the customer to cancel?",
                   "criteria": ["low", "medium", "high"]},
    "refund_requested": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
}

# body, department, urgency (0 not urgent, 1 soon, 2 critical), churn (0 low, 1 medium, 2 high), refund, language
EMAILS = [
    # --- English: billing ---
    ("Could you resend last month's invoice? No rush.", "billing", 0, 0, "no", "en"),
    ("I was charged twice for March. Please refund the duplicate charge.", "billing", 1, 1, "yes", "en"),
    ("Why was I billed in dollars instead of euros this time?", "billing", 1, 0, "no", "en"),
    ("Please update the VAT number on our invoices. Whenever you have time.", "billing", 0, 0, "no", "en"),
    ("You charged my card after I cancelled. I want my money back today.", "billing", 2, 2, "yes", "en"),
    ("I don't want a refund, I just need a corrected invoice for accounting.", "billing", 1, 0, "no", "en"),
    ("Our payment failed and the account says it will be suspended tomorrow. Please help!", "billing", 2, 1, "no", "en"),
    ("Can we switch from monthly to yearly billing? Not in a hurry.", "billing", 0, 0, "no", "en"),
    ("The invoice total doesn't match our quote. Please explain the difference.", "billing", 1, 1, "no", "en"),
    ("Refund the last payment please, we never used the product this month.", "billing", 1, 2, "yes", "en"),
    ("Can I get receipts for all payments this year? No rush at all.", "billing", 0, 0, "no", "en"),
    ("I'm not asking for my money back, just tell me why the price went up.", "billing", 1, 1, "no", "en"),
    ("Please stop charging my old card, it was stolen. Urgent.", "billing", 2, 0, "no", "en"),
    ("We were overcharged for 12 seats but only have 8 users. Please credit the difference.", "billing", 1, 1, "yes", "en"),
    ("Where do I download our invoices as PDF?", "billing", 0, 0, "no", "en"),
    ("Your pricing change doubled our bill. We expect a refund or we will leave.", "billing", 2, 2, "yes", "en"),
    ("Just confirming you received our bank transfer. No need to hurry.", "billing", 0, 0, "no", "en"),
    ("Can you add our PO number to future invoices?", "billing", 0, 0, "no", "en"),
    ("The discount code didn't apply at checkout. Can you fix the charge?", "billing", 1, 0, "no", "en"),
    ("Keep the money, I don't need a refund. I just want the invoice address fixed.", "billing", 0, 0, "no", "en"),
    ("I was billed for a plan I downgraded from last week.", "billing", 1, 1, "yes", "en"),
    ("Please cancel the auto-renewal before it charges us on Friday.", "billing", 1, 2, "no", "en"),
    ("Our finance team needs the invoices split per department. Whenever convenient.", "billing", 0, 0, "no", "en"),
    ("Why is there a late fee? We paid on time.", "billing", 1, 1, "no", "en"),
    ("I would like my money back for the unused months, please.", "billing", 1, 2, "yes", "en"),
    # --- English: technical ---
    ("The app crashes every time I log in and I have a demo in an hour!", "technical", 2, 1, "no", "en"),
    ("The login page returns an error 500.", "technical", 2, 0, "no", "en"),
    ("My reports won't download since yesterday's update.", "technical", 1, 0, "no", "en"),
    ("I don't want a refund, I just want the app to work.", "technical", 1, 1, "no", "en"),
    ("Videos in the app play without sound.", "technical", 1, 0, "no", "en"),
    ("Since the update I can't attach files to messages.", "technical", 1, 0, "no", "en"),
    ("Dark mode makes some buttons invisible. Not urgent, just letting you know.", "technical", 0, 0, "no", "en"),
    ("Our whole team is locked out. Production is down. Please call us now.", "technical", 2, 2, "no", "en"),
    ("The export to CSV puts all columns in one cell.", "technical", 1, 0, "no", "en"),
    ("Sync between phone and desktop stopped working three days ago.", "technical", 1, 1, "no", "en"),
    ("Nothing works after the update. This is the third outage this month; we're evaluating other tools.", "technical", 2, 2, "no", "en"),
    ("Small thing: a typo on the settings page. No rush.", "technical", 0, 0, "no", "en"),
    ("Password reset emails never arrive.", "technical", 2, 1, "no", "en"),
    ("The API returns 429 even though we are far below the rate limit.", "technical", 1, 1, "no", "en"),
    ("Two-factor codes are rejected as invalid.", "technical", 2, 1, "no", "en"),
    ("The calendar integration shows events one hour off.", "technical", 1, 0, "no", "en"),
    ("The app is so slow it's unusable. Fix it or refund us.", "technical", 2, 2, "yes", "en"),
    ("Search doesn't find documents older than a year.", "technical", 1, 0, "no", "en"),
    ("Notifications arrive twice. Not a big deal.", "technical", 0, 0, "no", "en"),
    ("After the update the mobile app shows a blank screen.", "technical", 2, 1, "no", "en"),
    ("I can't change my profile picture, the upload spins forever.", "technical", 0, 0, "no", "en"),
    ("Our webhook stopped firing last night and orders are piling up.", "technical", 2, 1, "no", "en"),
    ("Is there a known issue with the Android widget? It never refreshes.", "technical", 1, 0, "no", "en"),
    ("Printing a report cuts off the last column.", "technical", 0, 0, "no", "en"),
    ("The app logs me out every few minutes.", "technical", 1, 1, "no", "en"),
    # --- English: sales ---
    ("How much is the Business plan for 20 people?", "sales", 0, 0, "no", "en"),
    ("We'd like a demo of the new features next week.", "sales", 0, 0, "no", "en"),
    ("Is there a discount if we pay for three years upfront?", "sales", 0, 0, "no", "en"),
    ("Can we add more storage to our plan, and what does it cost?", "sales", 0, 0, "no", "en"),
    ("We need a quote for 50 licenses before our budget meeting on Monday.", "sales", 1, 0, "no", "en"),
    ("Do you offer a nonprofit discount?", "sales", 0, 0, "no", "en"),
    ("We're comparing you with two competitors. What makes your Enterprise plan better?", "sales", 1, 1, "no", "en"),
    ("Can we upgrade today? Our trial ends tonight.", "sales", 2, 0, "no", "en"),
    ("What's the price difference between Pro and Business?", "sales", 0, 0, "no", "en"),
    ("We want SSO. Which plan includes it?", "sales", 0, 0, "no", "en"),
    ("Our contract renews next month and we're considering switching unless the price drops.", "sales", 1, 2, "no", "en"),
    ("Do you have an education plan for a school of 300 students?", "sales", 0, 0, "no", "en"),
    ("Please send a proposal for the Enterprise tier.", "sales", 0, 0, "no", "en"),
    ("Can I try the premium features before buying?", "sales", 0, 0, "no", "en"),
    ("We need to add 10 seats before tomorrow's onboarding.", "sales", 2, 0, "no", "en"),
    ("Is there a reseller program in the Nordics?", "sales", 0, 0, "no", "en"),
    ("Could someone call me about volume pricing? No rush.", "sales", 0, 0, "no", "en"),
    ("We'd like to downgrade to the cheaper plan; the current one is too expensive.", "sales", 1, 2, "no", "en"),
    ("Does the annual plan include priority support?", "sales", 0, 0, "no", "en"),
    ("We are ready to buy but need a security questionnaire filled in first.", "sales", 1, 0, "no", "en"),
    # --- English: other ---
    ("Thanks, everything works great now!", "other", 0, 0, "no", "en"),
    ("Do you have an office in Malmö?", "other", 0, 0, "no", "en"),
    ("Where can I read your privacy policy?", "other", 0, 0, "no", "en"),
    ("Are you open on Saturdays?", "other", 0, 0, "no", "en"),
    ("I'd like to delete my account and all my data.", "other", 1, 2, "no", "en"),
    ("Can I change the email address on my account?", "other", 0, 0, "no", "en"),
    ("Your support agent was very helpful yesterday, thank you.", "other", 0, 0, "no", "en"),
    ("Do you have job openings for developers?", "other", 0, 0, "no", "en"),
    ("What is your company's postal address?", "other", 0, 0, "no", "en"),
    ("We received a legal request and need your data processing agreement today.", "other", 2, 0, "no", "en"),
    ("I'm writing an article about your product. Could I interview someone?", "other", 0, 0, "no", "en"),
    ("Please remove me from the newsletter.", "other", 0, 1, "no", "en"),
    ("I'm unhappy with how my complaint was handled and I'm cancelling.", "other", 1, 2, "no", "en"),
    ("Is your data stored in the EU?", "other", 0, 0, "no", "en"),
    ("Happy new year from all of us!", "other", 0, 0, "no", "en"),
    # --- English: ambiguous billing/technical ---
    ("The payment page crashes when I try to update my card.", "billing", 1, 1, "no", "en"),
    ("I can't download my invoice, the button gives an error.", "billing", 1, 0, "no", "en"),
    ("Checkout fails with an error, so I can't pay my bill.", "billing", 2, 1, "no", "en"),
    ("The billing page shows a blank screen after the update.", "technical", 1, 0, "no", "en"),
    ("My card was charged but the app still says my subscription expired.", "billing", 2, 1, "no", "en"),
    ("The app crashed during payment and I think I was charged twice. Refund the extra one.", "billing", 2, 1, "yes", "en"),
    ("The invoice PDF opens with garbled characters.", "technical", 0, 0, "no", "en"),
    ("Receipts show the wrong currency symbol in the app.", "technical", 0, 0, "no", "en"),
    ("After paying, my account still shows the free plan. Is that a bug?", "billing", 1, 1, "no", "en"),
    ("The payment form rejects every card I try.", "billing", 2, 1, "no", "en"),
    ("Our API integration double-billed customers because of your webhook retries.", "technical", 2, 2, "no", "en"),
    ("The pricing page won't load in Safari.", "technical", 0, 0, "no", "en"),
    ("I got an error mid-purchase; did the upgrade go through or not?", "billing", 1, 0, "no", "en"),
    ("The invoice email links are broken.", "technical", 1, 0, "no", "en"),
    ("The bill shows usage from a feature that crashed and never worked.", "billing", 1, 1, "yes", "en"),
    # --- Swedish (25) ---
    ("Kan ni skicka om förra månadens faktura? Inte bråttom.", "billing", 0, 0, "no", "sv"),
    ("Jag har debiterats två gånger för mars. Snälla återbetala den dubbla dragningen.", "billing", 1, 1, "yes", "sv"),
    ("Jag vill inte ha pengarna tillbaka, bara en fungerande app.", "technical", 1, 1, "no", "sv"),
    ("Appen kraschar varje gång jag loggar in och jag har en demo om en timme!", "technical", 2, 1, "no", "sv"),
    ("Kan jag få fakturan som PDF?", "billing", 0, 0, "no", "sv"),
    ("Appen kraschar när jag öppnar inställningarna.", "technical", 1, 0, "no", "sv"),
    ("Kan ni skicka en offert på 10 extra licenser?", "sales", 0, 0, "no", "sv"),
    ("Vilken är er postadress?", "other", 0, 0, "no", "sv"),
    ("Vad kostar Pro-planen för ett team på 20 personer?", "sales", 0, 0, "no", "sv"),
    ("Er uppdatering förstörde fakturaexporten. Tredje gången nu, vi tittar på andra verktyg.", "technical", 2, 2, "no", "sv"),
    ("Ni drog pengar efter att jag sagt upp avtalet. Jag vill ha tillbaka dem direkt.", "billing", 2, 2, "yes", "sv"),
    ("Jag vill inte ha någon återbetalning, rätta bara adressen på fakturan.", "billing", 0, 0, "no", "sv"),
    ("Ingen brådska, men kan ni lägga till vårt organisationsnummer på fakturorna?", "billing", 0, 0, "no", "sv"),
    ("Hela teamet är utelåst och produktionen står still. Ring oss nu.", "technical", 2, 2, "no", "sv"),
    ("Lösenordsåterställningen skickar aldrig något mejl.", "technical", 2, 1, "no", "sv"),
    ("Exporten till CSV lägger alla kolumner i en cell.", "technical", 1, 0, "no", "sv"),
    ("Finns det rabatt om vi betalar tre år i förväg?", "sales", 0, 0, "no", "sv"),
    ("Vi vill gärna ha en demo av de nya funktionerna.", "sales", 0, 0, "no", "sv"),
    ("Vårt avtal förnyas nästa månad och vi byter leverantör om priset inte sänks.", "sales", 1, 2, "no", "sv"),
    ("Tack, nu fungerar allt!", "other", 0, 0, "no", "sv"),
    ("Har ni kontor i Göteborg?", "other", 0, 0, "no", "sv"),
    ("Jag vill radera mitt konto och all min data.", "other", 1, 2, "no", "sv"),
    ("Betalsidan kraschar när jag försöker byta kort.", "billing", 1, 1, "no", "sv"),
    ("Appen kraschade under betalningen och jag tror att jag debiterades två gånger. Återbetala den extra.", "billing", 2, 1, "yes", "sv"),
    ("Kvittona visar fel valutasymbol i appen, inget brådskande.", "technical", 0, 0, "no", "sv"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "questions.yaml").write_text(yaml.safe_dump(QUESTIONS, allow_unicode=True, sort_keys=False),
                                        encoding="utf-8")
    levels = QUESTIONS["urgency"]["criteria"], QUESTIONS["churn_risk"]["criteria"]
    with open(OUT / "emails.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["body", "department", "urgency", "churn_risk", "refund_requested", "language"])
        for body, dept, urg, churn, refund, lang in EMAILS:
            w.writerow([body, dept, levels[0][urg], levels[1][churn], refund, lang])
    sv = sum(e[5] == "sv" for e in EMAILS)
    print(f"wrote {len(EMAILS)} emails ({sv} Swedish, {100 * sv / len(EMAILS):.0f}%) to {OUT}")


if __name__ == "__main__":
    main()
