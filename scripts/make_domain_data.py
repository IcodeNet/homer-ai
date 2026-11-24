# ruff: noqa: E501
"""Write the invented emails and service-desk tickets of chapter 10.

Run: uv run python scripts/make_domain_data.py          (writes the two files)
     uv run python scripts/make_domain_data.py --check  (exits 1 if they differ)

Everything here is made up. No real customer, employee or company wrote any of
it. The texts are built from the sentence lists below with a seeded random
number generator, so the same seed always gives the same files, byte for byte:

    data/domain/emails/emails.jsonl          400 emails to an invented savings firm
    data/domain/servicedesk/tickets.jsonl    200 tickets to an invented IT service desk

The split into training and test is made here, and it is made BY WORDING: about
a quarter of the sentences in each list are kept for the test set only. A model
is then tested on wordings it never trained on. Splitting the finished texts at
random would put near-copies of every training text into the test set.

Template text is still far easier than real mail (see data/domain/README.md).
"""

import argparse
import json
import random
import sys
from pathlib import Path

DOMAIN = Path(__file__).parent.parent / "data" / "domain"
EMAILS_FILE = DOMAIN / "emails" / "emails.jsonl"
TICKETS_FILE = DOMAIN / "servicedesk" / "tickets.jsonl"
SEED = 10

# ------------------------------------------------------------------ emails
PRODUCTS = [
    "Easy Access Saver",
    "1-Year Fixed Saver",
    "95-Day Notice Saver",
    "Cash ISA",
]
MONTHS = ["March", "May", "June", "August", "September", "October"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
AMOUNTS = ["250", "800", "1,500", "3,000", "5,000", "12,000"]
COUNTS = ["two", "three", "four", "five"]

GREETINGS = ["Hello,", "Hi,", "Good morning,", "Dear team,", "Dear Sir or Madam,", ""]
CLOSINGS = ["Thanks", "Kind regards", "Regards", "Many thanks", "Thank you", ""]

ENQUIRIES = [
    "Could you tell me the current rate on the {product}?",
    "How do I open a {product}? Is it all done online?",
    "I'd like to change the bank account linked to my {product}. What do you need?",
    "I have moved house. How do I update my address?",
    "When will the statement for my {product} be ready?",
    "Can I pay into my {product} by standing order?",
    "What is the most I can hold in the {product}?",
    "My {product} matures next month. What are my options?",
    "Can you confirm that my transfer of £{amount} arrived on {day}?",
    "I can't remember my password. How do I reset it?",
    "Do you offer joint accounts on the {product}?",
    "Please can you send me the form to transfer my ISA to you?",
    "Is there a charge for closing my {product}?",
    "How long does a withdrawal from the {product} take to reach my bank?",
    "I want to give notice on my {product}. How do I do that?",
    "Which documents do you need to register a power of attorney?",
    "Is the interest on the {product} paid monthly or yearly?",
    "I paid in £{amount} on {day}. When does it start to earn interest?",
    "Can I open a second {product} for my daughter, who is 19?",
    "Do you have an app, or is it only the website?",
]
PRAISE = [
    "Just a note to say thank you. My question about the {product} was sorted in minutes.",
    "The new app is much easier to use than the old one. Well done.",
    "Thank you for the quick reply about my {product}. Very helpful.",
    "I was impressed by how fast my withdrawal of £{amount} arrived.",
]
# They use the words a keyword list looks for, and are neither complaint nor vulnerable.
TRICKY = [
    "No complaints from me: the transfer into my {product} arrived the same day. Could you send a letter to confirm it?",
    "I'm not unhappy with my {product} at all, I only want to know if a better rate is coming.",
    "A friend was disappointed with her old bank and wants to move to you. How does she open a {product}?",
    "Where can I read your complaints procedure? I need it for a form. Nothing is wrong.",
    "I was worried the payment of £{amount} was lost, but it has arrived now, so no problem.",
    "I work nights at a hospital, so please phone me after 2pm about my {product}.",
    "My mother is in good health but wants me to help with her {product}. How do we set that up?",
    "I'm a carer by profession and my client has asked how the {product} works. Do you have a leaflet?",
    "My late payment last week was my own fault, not yours. Can you confirm the £{amount} has arrived now?",
    "I'd be disappointed to miss the new rate. When does the {product} offer end?",
    "My wife and I are both retired and in no hurry. What is the notice period on the {product}?",
    "Nobody could complain about that rate! How do I add £{amount} to my {product}?",
]
ISSUES = [
    "My withdrawal of £{amount} was promised for {day} and it still has not arrived.",
    "I gave notice on my {product} in {month} and now you say there is no record of it.",
    "You changed the rate on my {product} without telling me.",
    "I have been waiting {count} weeks for my ISA transfer to complete.",
    "I was on hold for over an hour and then the call was cut off.",
    "You sent my statement to my old address, even though I updated it in {month}.",
    "The interest paid on my {product} is lower than the rate I was quoted.",
    "Nobody has replied to the {count} messages I sent about my {product}.",
    "Your app has locked me out {count} times this month.",
    "I was told on the phone that closing my {product} was free, and now I have been charged.",
    "My deposit of £{amount} left my bank on {day} and is not showing in my {product}.",
    "You closed my {product} without asking me.",
    "The letter about my {product} had somebody else's name on it.",
    "I asked for a paper statement in {month} and have had nothing.",
    "My maturity instructions for the {product} were ignored.",
    "I keep getting emails meant for another customer.",
]
EXPLICIT = [
    "I want to make a formal complaint. {issue}",
    "I am very unhappy with your service. {issue}",
    "This is a complaint. {issue}",
    "{issue} I am extremely disappointed. Please treat this as a complaint.",
    "{issue} This is unacceptable.",
    "{issue} I am not happy about this at all.",
    "I wish to complain about the following. {issue}",
    "{issue} Frankly the service has been appalling.",
    "I am writing to complain. {issue}",
    "{issue} I am deeply dissatisfied with how this was handled.",
    "Please log this as a complaint: {issue}",
    "{issue} I am fed up and I want this looked at properly.",
]
# Dissatisfaction with none of the words a keyword list looks for.
IMPLICIT = [
    "{issue} Can somebody please explain how this happened?",
    "{issue} I have now asked {count} times.",
    "{issue} I expected better from you.",
    "{issue} What are you going to do to put this right?",
    "{issue} I shouldn't have to keep chasing.",
    "{issue} If this isn't sorted by {day} I will move my money elsewhere.",
    "{issue}",
    "{issue} This is not what I was told when I opened the account.",
    "{issue} I'm starting to wonder whether my money is safe with you.",
    "{issue} Who do I need to speak to so that this gets fixed?",
    "{issue} I have been a customer for years and this has never happened before.",
    "{issue} Please don't send me another standard reply.",
]
VULNERABLE = [
    "My husband passed away last month and I am dealing with everything on my own.",
    "My wife died recently and the account was in her name.",
    "I have just come out of hospital and can't get to a branch.",
    "I was diagnosed with cancer in {month} and I'm finding paperwork hard.",
    "I lost my job in {month} and money is very tight.",
    "I am registered disabled and cannot use the phone easily.",
    "I care full-time for my mother, who has dementia.",
    "My father is in a care home now and I am handling his money.",
    "I am recently bereaved and still sorting out the funeral.",
    "I have a serious illness and spend most weeks in treatment.",
    "I was made redundant in {month} and I have no income at present.",
    "I have a hearing impairment, so please do not phone me.",
]
# Signs of vulnerability with none of the words a keyword list looks for.
VULNERABLE_SOFT = [
    "I find your letters very confusing and my daughter usually reads them for me.",
    "I'm 84 and I don't use the internet.",
    "Things are very difficult at home at the moment and I can't cope with much.",
    "I need this money to pay my rent this week.",
    "My memory isn't what it was, so please put everything in writing.",
    "I've been signed off work with stress since {month}.",
    "English is not my first language, so please write simply.",
    "I'm behind on my bills and the letters keep coming.",
    "My eyesight is poor now and I can't read small print.",
    "I've had a very hard year since my divorce and I am not managing well.",
    "I don't really understand any of this. My son used to do it all for me.",
    "I can't sleep for worrying about money.",
]

EMAIL_POOLS = {
    "enquiry": ENQUIRIES,
    "praise": PRAISE,
    "tricky": TRICKY,
    "issue": ISSUES,
    "explicit": EXPLICIT,
    "implicit": IMPLICIT,
    "vulnerable": VULNERABLE,
    "vulnerable_soft": VULNERABLE_SOFT,
}
# (complaint, vulnerable) and how often, out of 100
EMAIL_MIX = [
    ((False, False), 42),
    ((True, False), 30),
    ((False, True), 13),
    ((True, True), 15),
]
EMAIL_COUNTS = {"train": 280, "val": 40, "test": 80}


def split_pool(items, rng, share=0.25):
    """(wordings for training, wordings kept for the test set), picked by `rng`."""
    held = set(rng.sample(range(len(items)), max(1, round(len(items) * share))))
    train = [item for i, item in enumerate(items) if i not in held]
    test = [item for i, item in enumerate(items) if i in held]
    return train, test


def fill(text, rng, product):
    return text.format(
        product=product,
        amount=rng.choice(AMOUNTS),
        day=rng.choice(DAYS),
        month=rng.choice(MONTHS),
        count=rng.choice(COUNTS),
    )


def one_email(rng, pools, complaint, vulnerable):
    """One email as (text, product or None, how each label shows in the words)."""
    product = rng.choice(PRODUCTS)
    styles = {"complaint": "none", "vulnerable": "none", "tricky": False}
    if complaint:
        styles["complaint"] = rng.choice(["explicit", "implicit"])
        frame = rng.choice(pools[styles["complaint"]])
        issue = fill(rng.choice(pools["issue"]), rng, product)
        core = fill(frame.replace("{issue}", issue), rng, product)
    else:
        kind = rng.choices(["enquiry", "praise", "tricky"], weights=[70, 8, 22])[0]
        if vulnerable:
            kind = "enquiry"
        styles["tricky"] = kind == "tricky"
        core = fill(rng.choice(pools[kind]), rng, product)
    parts = [core]
    if vulnerable:
        styles["vulnerable"] = rng.choice(["explicit", "soft"])
        pool = "vulnerable" if styles["vulnerable"] == "explicit" else "vulnerable_soft"
        sentence = fill(rng.choice(pools[pool]), rng, product)
        parts.insert(rng.choice([0, 1]), sentence)
    body = " ".join(parts)
    text = " ".join(p for p in (rng.choice(GREETINGS), body, rng.choice(CLOSINGS)) if p)
    if rng.random() < 0.15:
        text = text.lower()  # some people never use capitals
    return (
        text,
        (product if product in text or product.lower() in text else None),
        styles,
    )


def make_emails(seed=SEED):
    """The 400 emails, as a list of dicts in file order (training, validation, test)."""
    rng = random.Random(seed)
    train_pools, test_pools = {}, {}
    for name, items in EMAIL_POOLS.items():
        train_pools[name], test_pools[name] = split_pool(items, rng)
    emails, seen = [], set()
    for split, wanted in EMAIL_COUNTS.items():
        pools = test_pools if split == "test" else train_pools
        made = 0
        while made < wanted:
            complaint, vulnerable = rng.choices(
                [labels for labels, _ in EMAIL_MIX], weights=[w for _, w in EMAIL_MIX]
            )[0]
            text, product, styles = one_email(rng, pools, complaint, vulnerable)
            if text in seen:
                continue
            seen.add(text)
            made += 1
            emails.append(
                {
                    "id": f"e{len(emails) + 1:03d}",
                    "split": split,
                    "text": text,
                    "complaint": complaint,
                    "vulnerable": vulnerable,
                    "product": product,
                    "styles": styles,
                }
            )
    return emails


# ------------------------------------------------------------------ tickets
# (category, knowledge-base article or None, eight ways people say it)
TOPICS = [
    (
        "access",
        "kb-01",
        [
            "I can't log in, it says my password is wrong",
            "forgot my password again, can you reset it",
            "My password expired over the weekend and now I'm locked out",
            "The reset link never arrived in my inbox",
            "account locked after too many tries",
            "can't sign in to my laptop this morning",
            "need a new password, the old one stopped working",
            "It keeps saying invalid credentials",
        ],
    ),
    (
        "access",
        "kb-02",
        [
            "I got a new phone and the code app is gone",
            "authenticator app isn't giving me codes on my new mobile",
            "It asks for a six-digit code and I don't have my old phone any more",
            "how do I move the MFA to a different phone",
            "Lost my phone so I can't do the two-step sign in",
            "the app on my phone that makes the login numbers needs setting up again",
            "QR code for the authenticator won't scan",
            "second step of login fails since I changed handset",
        ],
    ),
    (
        "network",
        "kb-03",
        [
            "VPN won't connect from home",
            "Stuck on Connecting when I try the VPN",
            "can't reach the shared drives when working from home",
            "finance system won't open at home, works fine in the office",
            "the remote connection keeps timing out at my house",
            "VPN client asks me to sign in over and over",
            "Working from a hotel and can't get onto the company network",
            "home working: no access to the drives",
        ],
    ),
    (
        "network",
        "kb-04",
        [
            "wifi keeps dropping in meeting room 3",
            "can't join the Staff wifi on my laptop",
            "The wireless is really patchy on the second floor",
            "laptop connects to Guest but not Staff",
            "internet cuts out every few minutes at my desk",
            "no signal in the board room",
            "wi-fi asks for my login again and again",
            "Connection drops whenever I walk to the kitchen area",
        ],
    ),
    (
        "email",
        "kb-05",
        [
            "I can't see the accounts shared mailbox",
            "was given access to the team inbox yesterday but it isn't showing",
            "how do I add the invoices mailbox to my mail",
            "shared mailbox disappeared from my mail app",
            "Permission error when I open the HR inbox",
            "the group email box isn't in my folder list",
            "need the support inbox to appear on my laptop",
            "my colleague can see the shared inbox and I can't",
        ],
    ),
    (
        "email",
        "kb-06",
        [
            "mailbox full warning, can't send anything",
            "It says I've run out of space for emails",
            "my mail is over quota",
            "can't send, says storage limit reached",
            "how do I archive old emails",
            "Emails bounce back saying my inbox is full",
            "need more room in my mailbox",
            "getting a message that my mail storage is nearly used up",
        ],
    ),
    (
        "hardware",
        "kb-07",
        [
            "the printer on floor 2 shows offline",
            "can't print, the printer isn't in my list",
            "Print jobs just sit in the queue",
            "printer says offline for everyone here",
            "nothing comes out when I print",
            "how do I add the office printer to my laptop",
            "The copier by reception won't take my print jobs",
            "printing stopped working after I moved desks",
        ],
    ),
    (
        "hardware",
        "kb-08",
        [
            "my laptop is really slow today",
            "the fan is going non-stop and it's hot",
            "Laptop takes ten minutes to start up",
            "everything freezes when I have a few tabs open",
            "computer is crawling, can't work like this",
            "it's so sluggish since the last update",
            "laptop sounds like a jet engine",
            "My machine hangs every time I open a spreadsheet",
        ],
    ),
    (
        "software",
        "kb-09",
        [
            "I need the diagram tool installed",
            "can I get a licence for the PDF editor",
            "How do I install the screenshot program",
            "requesting the project planning software",
            "need the statistics package for a new project",
            "Please add the design app to my laptop",
            "it won't let me install a program I downloaded",
            "who approves a paid licence for the video editor",
        ],
    ),
    (
        "security",
        "kb-10",
        [
            "got a weird email asking for my password",
            "I think this message is a scam, what do I do with it",
            "An email from the 'CEO' wants gift cards urgently",
            "I clicked a link in a dodgy email, help",
            "suspicious invoice attachment from someone I don't know",
            "email says my account will be closed unless I log in now",
            "is this parcel delivery email real?",
            "Think I typed my password into a fake page",
        ],
    ),
    # No article covers these: a person has to pick them up.
    (
        "hardware",
        None,
        [
            "new starter on Monday needs a laptop and two screens",
            "can I have a second monitor",
            "my keyboard has stopped working",
            "screen is cracked after I dropped it",
            "docking station doesn't charge the laptop",
            "headset mic not working on calls",
            "need a replacement charger",
            "mouse is dead even with new batteries",
        ],
    ),
    (
        "software",
        None,
        [
            "the expenses system crashes when I attach a receipt",
            "spreadsheet program closes by itself",
            "video call app has no sound",
            "The timesheet app shows a blank page",
            "the customer database is giving an error 500",
            "calendar doesn't sync to my phone",
            "browser keeps crashing on the intranet",
            "report export fails every time",
        ],
    ),
    (
        "security",
        None,
        [
            "I left my laptop on the train",
            "my work phone was stolen last night",
            "found a USB stick in the car park, is it safe to plug in",
            "someone followed me through the door without a badge",
            "I think a colleague is using my login",
            "lost my door pass",
            "a visitor was left alone near the server room",
            "antivirus popped up a warning about a file",
        ],
    ),
    (
        "access",
        None,
        [
            "need access to the finance folder",
            "please add me to the marketing shared drive",
            "new joiner needs an account created",
            "can you give me edit rights on the team site",
            "leaver's account needs closing on Friday",
            "I've changed teams and need the sales folders",
            "contractor needs a temporary login",
            "can't open the board papers folder, access denied",
        ],
    ),
    (
        "network",
        None,
        [
            "the whole office has no internet",
            "video calls are choppy for everyone today",
            "can't reach the intranet from the north office",
            "network port at my desk is dead",
            "the meeting room screen won't connect to the network",
            "websites are very slow from the office",
            "our branch link has been down since 9",
            "need a network cable for my desk",
        ],
    ),
    (
        "email",
        None,
        [
            "an email I sent hasn't arrived",
            "I need a distribution list for the project",
            "emails from a client go to junk",
            "can't open an attachment",
            "calendar invites aren't arriving",
            "set up an out-of-office for a colleague who is off sick",
            "my signature has vanished",
            "need to recall an email I sent by mistake",
        ],
    ),
]
TICKET_OPENERS = [
    "",
    "",
    "Hi,",
    "Hello,",
    "Morning,",
    "Hi team,",
    "Urgent:",
    "Quick one:",
]
TICKET_ENDINGS = [
    "",
    "",
    "Thanks",
    "Please help.",
    "It's urgent, I have a client call at 2.",
    "This started this morning.",
    "Been like this since yesterday.",
    "Can someone call me?",
    "Not urgent.",
    "Same thing happened last week.",
]
TICKET_COUNTS = {"train": 150, "test": 50}


def make_tickets(seed=SEED):
    """The 200 tickets, as a list of dicts in file order (training, then test)."""
    rng = random.Random(seed)
    wordings = {"train": [], "test": []}
    for category, article, ways in TOPICS:
        train, test = split_pool(ways, rng)
        wordings["train"] += [(category, article, way) for way in train]
        wordings["test"] += [(category, article, way) for way in test]
    tickets, seen = [], set()
    for split, wanted in TICKET_COUNTS.items():
        made = 0
        while made < wanted:
            category, article, way = rng.choice(wordings[split])
            parts = (rng.choice(TICKET_OPENERS), way, rng.choice(TICKET_ENDINGS))
            text = " ".join(p for p in parts if p)
            if rng.random() < 0.2:
                text = text.lower()
            if text in seen:
                continue
            seen.add(text)
            made += 1
            tickets.append(
                {
                    "id": f"t{len(tickets) + 1:03d}",
                    "split": split,
                    "text": text,
                    "category": category,
                    "kb": article,
                }
            )
    return tickets


def to_jsonl(rows):
    """One JSON object per line; the exact text that is written to the file."""
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare, write nothing")
    args = parser.parse_args()
    wrong = 0
    for path, rows in ((EMAILS_FILE, make_emails()), (TICKETS_FILE, make_tickets())):
        text = to_jsonl(rows)
        if args.check:
            same = path.exists() and path.read_text(encoding="utf-8") == text
            wrong += not same
            print(
                f"{path.relative_to(DOMAIN.parent.parent)}  {'ok' if same else 'DIFFERENT'}"
            )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"{path.relative_to(DOMAIN.parent.parent)}  {len(rows)} rows")
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
