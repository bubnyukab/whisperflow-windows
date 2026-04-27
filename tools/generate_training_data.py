"""Generate synthetic (dirty_transcript, clean_transcript) pairs for fine-tuning."""

import json
import random
import re
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path(__file__).parent / "training_data.jsonl"
TARGET_PAIRS = 5000

# ── Filler words ───────────────────────────────────────────────────────────────
FILLERS = [
    "um", "uh", "like", "you know", "so", "basically", "literally",
    "kind of", "I mean", "right", "actually", "sort of",
]

# ── Spelling error map: (regex_pattern, replacement) ──────────────────────────
# Applied before lowercasing so case-sensitive patterns work.
SPELLING_SUBS = [
    (r"\bdefinitely\b",  "definately"),
    (r"\bDefinitely\b",  "Definately"),
    (r"\bdefinite\b",    "definate"),
    (r"\breceive\b",     "recieve"),
    (r"\bReceive\b",     "Recieve"),
    (r"\breceived\b",    "recieved"),
    (r"\bReceived\b",    "Recieved"),
    (r"\boccurred\b",    "occured"),
    (r"\boccurrence\b",  "occurence"),
    (r"\bseparate\b",    "seperate"),
    (r"\bSeparate\b",    "Seperate"),
    (r"\bseparately\b",  "seperately"),
    (r"\bthe\b",         "teh"),
    (r"\bThe\b",         "Teh"),
    (r"\bbecause\b",     "becuz"),
    (r"\bBecause\b",     "Becuz"),
    (r"\bI'm\b",         "im"),
    (r"\bdon't\b",       "dont"),
    (r"\bDon't\b",       "Dont"),
    (r"\bcan't\b",       "cant"),
    (r"\bCan't\b",       "Cant"),
    (r"\bwon't\b",       "wont"),
    (r"\bWon't\b",       "Wont"),
    (r"\bI'll\b",        "ill"),
    (r"\bthat's\b",      "thats"),
    (r"\bThat's\b",      "Thats"),
    (r"\bI've\b",        "ive"),
    (r"\btheir\b",       "thier"),
    (r"\bweird\b",       "wierd"),
    (r"\bWeird\b",       "Wierd"),
    (r"\bbeautiful\b",   "beatiful"),
    (r"\bBeautiful\b",   "Beatiful"),
    (r"\baccommodate\b", "accomodate"),
    (r"\bcommittee\b",   "commitee"),
    (r"\bCommittee\b",   "Commitee"),
    (r"\bexistence\b",   "existance"),
    (r"\bconsistent\b",  "consistant"),
    (r"\bpersistent\b",  "persistant"),
]
_COMPILED_SUBS = [(re.compile(p), r) for p, r in SPELLING_SUBS]


# ── Base clean sentences by category ──────────────────────────────────────────

WORK_SENTENCES = [
    "Please send me the report by end of day Friday.",
    "I need to schedule a meeting with the marketing team.",
    "The quarterly results exceeded our expectations by fifteen percent.",
    "Can you review this document before the presentation tomorrow?",
    "I'll follow up with the client after the call.",
    "We need to finalize the budget proposal by next week.",
    "The project deadline has been moved to the end of the month.",
    "Please add this item to the agenda for Thursday's meeting.",
    "I've attached the invoice for your review.",
    "We should discuss the new hiring requirements with HR.",
    "The stakeholders want an update on the current sprint.",
    "Can you send me the latest version of the slide deck?",
    "I need to reschedule my one-on-one with my manager.",
    "The client approved the proposal this morning.",
    "Please make sure the conference room is booked for two hours.",
    "We're looking for ways to reduce operational costs this quarter.",
    "The legal team needs to review the contract before we sign.",
    "I want to set up a kickoff call with the new vendor.",
    "Could you prepare a summary of last week's performance metrics?",
    "The team exceeded the sales target by twenty percent this month.",
    "I need to update the project roadmap before the board meeting.",
    "Please share the customer feedback from the last survey.",
    "We need to onboard three new employees next week.",
    "The product launch is scheduled for the first of next month.",
    "Can you take notes during today's all-hands meeting?",
    "I'll send the meeting invite for the strategy session.",
    "We should align on priorities before the end of the quarter.",
    "The budget for Q3 has been approved by finance.",
    "Please follow up with the vendor about the delivery timeline.",
    "I need a brief overview of the competitive landscape.",
    "The conference call is scheduled for three o'clock Eastern.",
    "Can you draft a response to the client's feedback?",
    "We need to update the SLA agreement before renewal.",
    "I'll circulate the agenda before tomorrow's meeting.",
    "The annual performance reviews start next Monday.",
    "Please make sure all action items are documented.",
    "I need to prepare talking points for the executive presentation.",
    "The team completed the project two days ahead of schedule.",
    "Can you set up a recurring weekly sync with the design team?",
    "We need to finalize the org chart before the announcement.",
    "Please update the project status in the shared spreadsheet.",
    "I want to discuss the rollout plan before we go live.",
    "The investor presentation needs to be ready by Wednesday.",
    "Can you compile the list of open action items from last sprint?",
    "We're moving to a new ticketing system starting next quarter.",
    "Please block two hours on my calendar for deep work tomorrow.",
    "The compliance audit is scheduled for next Tuesday.",
    "I need to get sign-off from the VP before proceeding.",
    "Can you send the onboarding materials to the new hire?",
    "We should review the contract terms with our legal advisor.",
    "The release notes need to be approved before deployment.",
    "Please coordinate with the events team for the off-site.",
    "I'll send a follow-up email to confirm the details.",
    "The budget variance report is due at the end of the week.",
    "Can you set up a demo environment for the client visit?",
    "We need to address the bottleneck in the approval workflow.",
    "Please reach out to the IT team about the network outage.",
    "I'd like to schedule a retrospective after the project closes.",
    "The new policy goes into effect at the start of next month.",
    "Can you update the team on the changes to the release plan?",
    "We should document the lessons learned from this project.",
    "Please ensure the presentation is saved in the shared drive.",
    "I need to review the time-off requests before Friday.",
    "The monthly newsletter should go out by Thursday morning.",
    "Can you help me prepare the agenda for the planning session?",
    "We're launching the new feature in the next sprint.",
    "Please update the contact list with the new account manager.",
    "I need to send a proposal to three potential vendors.",
    "The team will present the findings at next week's all-hands.",
    "We need to review the key performance indicators for this quarter.",
    "The client meeting is confirmed for Tuesday at ten in the morning.",
    "Please prepare the financial projections for the next two quarters.",
]

CASUAL_SENTENCES = [
    "I went to the grocery store after work today.",
    "Have you seen that new show on Netflix?",
    "We should grab coffee sometime this week.",
    "I can't believe how fast the weekend went by.",
    "My cat knocked over my coffee this morning.",
    "Did you catch the game last night?",
    "I'm thinking about redecorating my living room.",
    "The weather has been really nice this week.",
    "I tried that new restaurant downtown and it was amazing.",
    "I'm so tired, I stayed up way too late last night.",
    "We're planning a road trip for the long weekend.",
    "I finally finished that book I've been reading for months.",
    "The kids had a great time at the birthday party.",
    "I need to find a good plumber to fix the leak.",
    "Have you tried the new coffee shop on Fifth Street?",
    "I'm going to the gym after this, want to join?",
    "My phone battery dies so quickly lately.",
    "We should plan a get-together soon.",
    "I ordered takeout because I didn't feel like cooking.",
    "The traffic was terrible this morning.",
    "I just got back from visiting my family.",
    "Do you want to watch a movie tonight?",
    "I've been really into hiking lately.",
    "The kids are already on summer break.",
    "I need to get my car serviced soon.",
    "We went to the farmer's market on Saturday.",
    "I can't wait for the holidays to come.",
    "I finally cleaned out my closet this weekend.",
    "Have you heard about the new park they're building?",
    "I've been trying to cook more at home.",
    "My neighbor got a new puppy and it's adorable.",
    "I'm thinking of taking a cooking class.",
    "We had the most amazing pizza last Friday.",
    "I need to return these shoes, they don't fit.",
    "I binge-watched the whole season in one weekend.",
    "My back has been killing me from sitting all day.",
    "We're thinking of adopting a rescue dog.",
    "I finally got around to fixing the fence.",
    "The sunset last night was absolutely gorgeous.",
    "I got a great deal on flights for our vacation.",
    "My sister is visiting from out of town next week.",
    "I've been really enjoying the cooler weather.",
    "We're renovating the kitchen this summer.",
    "I can't believe how expensive groceries have gotten.",
    "I've started learning to play the guitar.",
    "We had a barbecue with the neighbors last weekend.",
    "I'm looking forward to the long weekend.",
    "My morning run felt really good today.",
    "I need to find a babysitter for Friday night.",
    "We're going to visit the botanical gardens on Sunday.",
    "I've been trying to drink more water every day.",
    "My commute was so smooth today for once.",
    "I finally signed up for that yoga class.",
    "The library is having a book sale this weekend.",
    "We're throwing a small dinner party next Saturday.",
    "I just discovered a great new podcast.",
    "I picked up a great book at the used bookstore.",
    "We're going camping for the first time this year.",
    "I've been trying to get better at budgeting.",
    "My plants are finally starting to grow.",
    "I need to call my mom, I haven't talked to her in a week.",
    "We found the best little taco place around the corner.",
    "I've been sleeping so much better since I got a new mattress.",
    "I forgot my umbrella and got completely soaked.",
    "We're thinking of getting a bigger apartment.",
    "I've been baking bread every weekend lately.",
    "The concert was so loud but absolutely worth it.",
    "I need a vacation, I'm completely burned out.",
    "We drove two hours to try that famous barbecue place.",
    "I finally started meditating and it's actually helping.",
]

QUESTION_SENTENCES = [
    "What time does the pharmacy close on Sundays?",
    "Can you tell me where the nearest gas station is?",
    "What is five times eight?",
    "How do you make a perfect omelette?",
    "What is the capital of Australia?",
    "How long does it take to fly from New York to London?",
    "What is the square root of one hundred and forty-four?",
    "Who wrote the novel Pride and Prejudice?",
    "What is twelve multiplied by fifteen?",
    "How do you say hello in Japanese?",
    "What is the boiling point of water in Celsius?",
    "How many days are there in a leap year?",
    "What is three hundred divided by twelve?",
    "Who was the first person to walk on the moon?",
    "What is the largest planet in our solar system?",
    "How many inches are in a foot?",
    "What year did World War Two end?",
    "What is seven to the power of three?",
    "How do you convert Fahrenheit to Celsius?",
    "What is the population of Canada?",
    "Who invented the telephone?",
    "What is one thousand minus three hundred and seventy-five?",
    "How many ounces are in a pound?",
    "What is the chemical symbol for gold?",
    "How do you remove a wine stain from a carpet?",
    "What is the speed of light in kilometers per second?",
    "Who painted the Mona Lisa?",
    "What is fifty percent of two hundred and forty?",
    "How many teaspoons are in a tablespoon?",
    "What is the tallest mountain in the world?",
    "What is nine times nine?",
    "How long does it take for the Earth to orbit the Sun?",
    "What is the sum of twenty-seven and thirty-eight?",
    "How do you get red wine out of white fabric?",
    "What is the largest ocean on Earth?",
    "How many grams are in a kilogram?",
    "What year was the Eiffel Tower built?",
    "What is four hundred minus one hundred and sixty-three?",
    "Who invented the World Wide Web?",
    "What is the fastest land animal?",
    "How do you make sourdough bread from scratch?",
    "What is six times seven?",
    "How many bones are in the human body?",
    "What is the deepest lake in the world?",
    "Who wrote Romeo and Juliet?",
    "What is two to the power of ten?",
    "How many feet are in a mile?",
    "What is the hardest natural substance on Earth?",
    "How do you calculate the area of a circle?",
    "What is the distance from the Earth to the Moon?",
    "Who discovered penicillin?",
    "What is eight times eleven?",
    "How many continents are there on Earth?",
    "What is the smallest country in the world?",
    "What is one hundred and twenty-five divided by five?",
    "What is the atomic number of carbon?",
    "How many planets are in our solar system?",
    "What is the square root of two hundred and twenty-five?",
    "Who wrote the Harry Potter series?",
    "What is twenty-four times three?",
    "How long does it take to boil an egg?",
    "What is the average weight of the human brain?",
    "Can you explain how a vaccine works?",
    "What is the difference between RAM and storage?",
    "How do you find the perimeter of a rectangle?",
    "What is the freezing point of water in Fahrenheit?",
    "How many calories are in a cup of cooked rice?",
    "What is forty-two divided by six?",
    "How do you reset a password on an iPhone?",
    "What is the longest river in Africa?",
]

COMMAND_SENTENCES = [
    "Write me an email to the team about the deadline change.",
    "Remind me to call the dentist tomorrow at nine.",
    "Set a timer for twenty minutes.",
    "Add milk and eggs to my grocery list.",
    "Turn off the lights in the living room.",
    "Send a text to John saying I'll be five minutes late.",
    "Create a new folder called Projects on my desktop.",
    "Play some relaxing music.",
    "Book a table for two at the Italian restaurant on Saturday.",
    "Set my alarm for six thirty in the morning.",
    "Draft an email apologizing to the client for the delay.",
    "Add a reminder to review the budget report on Thursday.",
    "Search for the best hiking trails near Denver.",
    "Call my sister.",
    "Start a new document and title it Meeting Notes.",
    "Schedule a meeting with the design team for next Wednesday.",
    "Look up the opening hours for the downtown library.",
    "Send a birthday message to my mother.",
    "Open the spreadsheet from last month.",
    "Order more printer paper online.",
    "Find the fastest route to the airport.",
    "Compose an out-of-office reply for next week.",
    "Add sunscreen and bug spray to the shopping list.",
    "Summarize the key points from today's meeting.",
    "Turn down the thermostat by two degrees.",
    "Create a to-do list for tomorrow.",
    "Schedule a haircut appointment for Saturday morning.",
    "Translate this paragraph to Spanish.",
    "Look up the weather forecast for the next five days.",
    "Write a thank-you note to the team after the project.",
    "Set a recurring reminder every Monday at eight AM.",
    "Find a recipe for chicken tikka masala.",
    "Open my calendar and show me next week.",
    "Draft a message to the landlord about the broken faucet.",
    "Send the report to my manager before three PM.",
    "Create a new playlist for my workout.",
    "Remind me to take my medication at nine PM.",
    "Find the contact number for the local vet.",
    "Book a flight to Seattle for next Friday.",
    "Write a short bio for my LinkedIn profile.",
    "Add this task to the sprint backlog.",
    "Set a meeting with HR for Tuesday afternoon.",
    "Compile a list of all open bugs in the system.",
    "Create a weekly report template in the shared drive.",
    "Find the status of my package delivery.",
    "Draft an agenda for Monday's team meeting.",
    "Add the conference call details to my calendar.",
    "Write a professional response to the customer complaint.",
    "Order dinner for the team meeting tonight.",
    "Start recording a voice memo.",
    "Send the updated contract to the vendor.",
    "Look up the definition of the word serendipity.",
    "Create a backup of my important files.",
    "Find me a good podcast about personal finance.",
    "Write the meeting summary and share it with the team.",
    "Set a focus session for ninety minutes with no interruptions.",
    "Book a hotel in Chicago for the weekend of the fifteenth.",
    "Draft a proposal for the new marketing campaign.",
    "Add olive oil and garlic to the shopping list.",
    "Remind me to water the plants on Friday evening.",
    "Look up the train schedule from Boston to New York.",
    "Cancel my gym membership and send the cancellation email.",
    "Create a shared document for the team retrospective notes.",
    "Find the best-rated electricians in my area.",
    "Set a three-hour work block every morning this week.",
    "Write a polite follow-up to the unanswered job application.",
    "Add the new team member to the Slack channel.",
    "Schedule a dentist appointment for sometime next month.",
]

TECHNICAL_SENTENCES = [
    "The API endpoint returns a JSON response with a status code of two hundred.",
    "We need to refactor the authentication module to use JWT tokens.",
    "The database query is taking too long due to a missing index.",
    "Deploy the new Docker container to the staging environment.",
    "The memory leak in the background worker was traced to a circular reference.",
    "We should add unit tests for the payment processing module.",
    "The CI/CD pipeline failed because of a failing linter check.",
    "Update the npm packages to fix the security vulnerability.",
    "The frontend is built with React and the backend uses FastAPI.",
    "We need to implement rate limiting on the public API endpoints.",
    "The SSL certificate expires in thirty days and needs to be renewed.",
    "I found a race condition in the concurrent file upload handler.",
    "The microservice communicates over gRPC with protocol buffers.",
    "We should migrate the database to a managed cloud service.",
    "The new feature branch needs to be rebased onto main before merging.",
    "The regex pattern is not matching emails with plus signs in them.",
    "We need to add caching to the product catalog endpoint.",
    "The Kubernetes pod keeps crashing due to an out-of-memory error.",
    "The TypeScript compiler is throwing a type error on line forty-two.",
    "We should use a message queue to decouple the two services.",
    "The function has a time complexity of O of n squared.",
    "We need to write a migration script for the schema change.",
    "The load balancer is not distributing traffic evenly across the pods.",
    "I'm implementing the binary search algorithm for the new feature.",
    "We should enable HTTP/2 on the web server for better performance.",
    "The GraphQL resolver is returning null for the nested user object.",
    "Add error handling for the edge case where the input is an empty string.",
    "The OAuth flow redirects to the wrong callback URL in production.",
    "We're using Redux for global state management in the React app.",
    "The build artifact is too large because of the unminified JavaScript.",
    "I need to set up environment variables for the staging configuration.",
    "The WebSocket connection drops after thirty seconds of inactivity.",
    "The integration test is failing because the mock server is not starting.",
    "We should add logging to the data transformation pipeline.",
    "The Python script requires version three point eleven or later.",
    "The git merge resulted in a conflict in the config file.",
    "We need to optimize the SQL join that is scanning the entire table.",
    "The component is not re-rendering when the parent state changes.",
    "I set up a webhook to trigger the deployment on every push to main.",
    "The Lambda function times out when processing large files.",
    "We should encrypt the sensitive fields in the database.",
    "I need to update the Dockerfile to use a smaller base image.",
    "The A/B test results showed a twelve percent improvement in conversion.",
    "We should version the API so we can deprecate old endpoints gracefully.",
    "The user interface needs to handle the loading and error states.",
    "I traced the bug to a null pointer dereference in the parser.",
    "The server is hitting a hundred percent CPU usage under load.",
    "We need to add pagination to the list endpoint to avoid timeouts.",
    "The pull request was approved after addressing the reviewer's comments.",
    "I need to write a script to automate the data cleanup process.",
    "The new endpoint requires OAuth two point zero authentication.",
    "The application logs are rotated daily and retained for thirty days.",
    "We should add a feature flag to gradually roll out the new dashboard.",
    "I'm using async and await to handle the asynchronous database calls.",
    "The search index needs to be rebuilt after the schema migration.",
    "We should document the API changes in the changelog before releasing.",
    "The cloud storage bucket needs proper access control policies.",
    "The benchmark shows the new algorithm is three times faster.",
    "We're using a binary heap for the priority queue implementation.",
    "The containerized service is deployed across three availability zones.",
    "I need to profile the code to find which function is causing the slowdown.",
    "The test coverage is currently at sixty-eight percent and needs to improve.",
    "We should add type hints to all the public functions in this module.",
    "The parser fails when the input contains special Unicode characters.",
    "I configured the reverse proxy to handle SSL termination at the edge.",
    "The new version of the library introduced a breaking change in the API.",
    "We need to set up monitoring and alerting for the new service.",
]

RUNON_SENTENCES = [
    "I went to the store and I bought some milk and eggs and bread and I forgot the butter so I had to go back and then the parking was terrible and I was late for everything.",
    "We had a meeting this morning and we talked about the project timeline and the budget and the team capacity and then we realized we needed to bring in two more developers and the deadline was moved.",
    "The software is running slowly because the database is not indexed properly and the queries are doing full table scans and the cache is not configured and the memory usage keeps climbing.",
    "I need to send an email to the client and prepare the slides and review the contract and schedule the follow-up call and make sure the team has all the information they need before Thursday.",
    "She told me that the meeting was moved and the agenda changed and there was a new stakeholder joining and the format would be a workshop style instead of a presentation which meant we had to redo all our materials.",
    "The new employee started on Monday and she met the team and set up her laptop and went through onboarding and had lunch with her manager and by the end of the day she had already submitted her first pull request.",
    "I downloaded the app and created an account and set my preferences and linked my calendar and imported my contacts and it still crashed every time I tried to open the settings menu.",
    "He worked on the feature for two days and wrote tests and got code review and fixed the review comments and merged to main and deployed to staging and it worked perfectly there but broke in production.",
    "We researched three different vendors and compared their pricing and read their documentation and ran a proof of concept and presented the results to the team and still couldn't agree on which one to choose.",
    "The report needs the sales figures from last quarter and the customer satisfaction scores and the churn data and the revenue breakdown and it all needs to be formatted for the executive summary by Friday morning.",
    "I cooked dinner and cleaned the kitchen and helped the kids with homework and got them ready for bed and then finally sat down to relax and realized it was already eleven o'clock.",
    "We launched the campaign and monitored the metrics and saw a spike in traffic and the servers started struggling and we had to scale up quickly and then the ad budget ran out before we expected.",
    "He explained that the system uses microservices and each service has its own database and they communicate through a message bus and the whole thing is deployed on Kubernetes across three availability zones.",
    "I need to finish the presentation and practice it and print handouts and get to the venue early and set up the projector and greet the guests and still be ready to present at nine sharp.",
    "The client called and said they were unhappy with the deliverable and wanted changes and could we meet on short notice and also the timeline needed to move up by two weeks which we had not anticipated at all.",
    "We updated the library and the build broke and we rolled back and the rollback failed and we had to restore from a backup and the whole thing took three hours on a Friday afternoon.",
    "She learned Python and then JavaScript and then she picked up React and then started studying system design and now she is preparing for technical interviews at top companies and getting callbacks.",
    "I called the doctor and made an appointment and then called the insurance and checked my coverage and then called the specialist and got a referral and the whole process took most of the morning.",
    "The flight was delayed by two hours and then the connection was missed and the airline rebooked us on the next available flight which was six hours later and we arrived at midnight instead of six PM.",
    "We signed up for the conference and registered three attendees and booked the hotel and arranged travel and submitted three talk proposals and two of them were accepted which was a great result.",
    "I finished the draft and sent it to my colleague for review and she made edits and sent it back and I revised it again and sent it to the manager and he had more changes and it went back and forth four times.",
    "The deployment went fine in the morning but by afternoon we got reports of errors and we checked the logs and found a database timeout and traced it to a slow query introduced in the latest release.",
    "We planned the sprint and assigned tickets and estimated points and set the goal and then on day two a critical bug came in from production and it consumed two days of the team's capacity.",
    "I signed up for the course and completed the first three modules and then got busy at work and missed a week and then it felt too hard to catch up so I started from the beginning again.",
    "The team built the feature in two weeks and it passed QA and went to UAT and the business approved it and it was deployed to production and within an hour two users reported that it wasn't working correctly.",
    "I woke up late and skipped breakfast and missed my train and had to take a taxi which was expensive and arrived twenty minutes into the meeting and my laptop battery was dead so I borrowed a charger.",
    "She applied for the job and got a phone screen and passed the technical interview and did the final round and waited three weeks to hear back and then got an offer below what she expected.",
    "The team refactored the module and updated the tests and the coverage went up but then the nightly build started failing because of an environment issue that had nothing to do with the refactor.",
    "I need to renew my passport and get a visa and book flights and arrange a pet sitter and pack and make sure all the bills are set to autopay before I leave for the trip next month.",
    "He started the meeting by reviewing last week's action items and then went through the new requirements and then asked for status updates and the whole thing ran forty minutes over the scheduled time.",
]

PROPER_NOUN_SENTENCES = [
    "I have a meeting with Sarah Johnson from the marketing department at two o'clock.",
    "We're using Microsoft Teams for our video calls and Slack for messaging.",
    "John sent me the report from the London office this morning.",
    "The conference is being held at the Marriott in downtown Chicago.",
    "Emily and David are leading the new product initiative.",
    "We need to finalize the contract with Amazon Web Services.",
    "Please send the files to Rebecca Torres before the deadline.",
    "The team in Sydney is ahead of us by fifteen hours.",
    "Microsoft announced a major update to Windows at their Build conference.",
    "I spoke with Dr. Patel at the Stanford Medical Center last week.",
    "Google released a new version of Chrome yesterday.",
    "The CEO, James Williams, will address the company on Friday.",
    "We have offices in New York, Berlin, and Singapore.",
    "Please CC Jennifer Lee on all future correspondence.",
    "The Salesforce integration is finally working as expected.",
    "Tom and Maria completed the audit for the Paris branch.",
    "Apple's new product announcement is scheduled for September.",
    "The project is funded by the National Science Foundation.",
    "I interviewed at Netflix last week and got a callback.",
    "Can you forward this to Alex Kim and the rest of the leadership team?",
    "We're migrating from AWS to Google Cloud by the end of the year.",
    "The legal review is being handled by Claire at Wilson and Associates.",
    "The founder, Michael Chen, started the company in his garage in 2012.",
    "The San Francisco office handles all West Coast client accounts.",
    "We're partnering with Shopify for the e-commerce integration.",
    "Please reach out to William from the Beijing team for the translation.",
    "OpenAI released a new model that outperformed the previous benchmark.",
    "I'll be at the Hilton in Boston from Tuesday through Thursday.",
    "Linda and Frank from the Denver office will join the call.",
    "We use Jira for project tracking and Confluence for documentation.",
    "The new hire, Priya Sharma, starts on the fifteenth of next month.",
    "I need to submit the expense report to Brian in accounting.",
    "The partnership with Toyota is set to be announced next quarter.",
    "Our CTO, Rachel Green, is presenting at the AWS conference.",
    "Please coordinate with Ivan from the Moscow team on the timeline.",
    "The Tokyo office operates on Japan Standard Time, nine hours ahead.",
    "Spotify and Apple Music are the two platforms we support.",
    "I sent the brief to Sophie and Marcus for their approval.",
    "The new data center is being built in Phoenix, Arizona.",
    "GitHub Copilot is now integrated into our development workflow.",
    "Sarah and Omar from the Dubai office will visit next month.",
    "The merger with DataCorp is pending regulatory approval.",
    "Please send the NDA to Kevin at Harrison Legal Group.",
    "We're attending the Consumer Electronics Show in Las Vegas in January.",
    "I need to reconnect with Dr. Ahmed from the WHO task force.",
    "Our engineering team uses Python, Go, and TypeScript.",
    "The regional manager, Patricia Brown, oversees twelve territories.",
    "We just signed a deal with Samsung for hardware integration.",
    "Natalie from investor relations will handle the press release.",
    "I have a call with the Stripe team to discuss payment processing.",
    "The Singapore branch is led by David Tan and his team of forty.",
    "We're rolling out Okta for single sign-on across all our tools.",
    "Robert and Christine from legal need to sign off on the terms.",
    "I heard back from Elena at the European Patent Office this morning.",
    "Our biggest clients include Boeing, Ford, and Johnson and Johnson.",
    "The DevOps conference in Seattle is in two weeks.",
    "We need to loop in Anthony from the Chicago office on this decision.",
    "The Zurich team is handling the European compliance requirements.",
    "I connected with Lisa from Deloitte at the finance summit last week.",
    "Please add Hannah Wright to the distribution list for weekly updates.",
    "The contract with Accenture expires at the end of Q3 this year.",
    "Carlos from the Mexico City office will lead the Latin America rollout.",
    "We presented the roadmap to the board and Mark from Goldman asked the hardest questions.",
    "The Dublin engineering hub is hiring ten senior developers this quarter.",
    "Grace and Leo from the Seoul team built the localization pipeline.",
    "I need to confirm the venue with the Westin before the invitations go out.",
    "Amazon just announced a new feature that competes directly with our product.",
    "Please check with Lena in Berlin before we commit to the timeline.",
]

ALL_SENTENCES: list[str] = (
    WORK_SENTENCES + CASUAL_SENTENCES + QUESTION_SENTENCES +
    COMMAND_SENTENCES + TECHNICAL_SENTENCES + RUNON_SENTENCES +
    PROPER_NOUN_SENTENCES
)


# ── Corruption functions ───────────────────────────────────────────────────────

def insert_fillers(text: str) -> str:
    """Randomly insert filler words between words and optionally at the start."""
    words = text.split()
    if len(words) < 3:
        return text
    result: list[str] = []
    for i, word in enumerate(words):
        result.append(word)
        # Insert a filler between words (not at the very end)
        if 0 < i < len(words) - 1 and random.random() < 0.17:
            result.append(random.choice(FILLERS))
    if random.random() < 0.28:
        result.insert(0, random.choice(FILLERS))
    return " ".join(result)


def remove_punctuation(text: str) -> str:
    """Strip common punctuation marks (STT often omits them)."""
    return re.sub(r"[.,?!;:]", "", text)


def apply_spelling_errors(text: str) -> str:
    """Replace correct words with common misspellings at random."""
    for pattern, replacement in _COMPILED_SUBS:
        if pattern.search(text) and random.random() < 0.40:
            text = pattern.sub(replacement, text, count=1)
    return text


def duplicate_words(text: str) -> str:
    """Randomly duplicate individual words to simulate audio glitches."""
    words = text.split()
    if len(words) < 4:
        return text
    result: list[str] = []
    for word in words:
        result.append(word)
        if random.random() < 0.045:
            result.append(word)
    return " ".join(result)


def corrupt(text: str) -> str:
    """Apply a random mix of corruptions to simulate real STT output."""
    do_fillers = random.random() < 0.72
    do_lower   = random.random() < 0.60
    do_punct   = random.random() < 0.58
    do_spell   = random.random() < 0.44
    do_dup     = random.random() < 0.18

    # Guarantee at least one corruption so input != output
    if not any([do_fillers, do_lower, do_punct, do_spell, do_dup]):
        do_fillers = True

    result = text
    # Spelling errors before lowercasing so case-sensitive patterns match
    if do_spell:
        result = apply_spelling_errors(result)
    if do_fillers:
        result = insert_fillers(result)
    if do_lower:
        result = result.lower()
    if do_punct:
        result = remove_punctuation(result)
    if do_dup:
        result = duplicate_words(result)

    return result


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_pairs(target: int) -> list[dict]:
    pairs: list[dict] = []
    max_attempts = target * 12
    attempts = 0
    while len(pairs) < target and attempts < max_attempts:
        clean = random.choice(ALL_SENTENCES)
        dirty = corrupt(clean)
        if dirty.strip() != clean.strip():
            pairs.append({"input": dirty, "output": clean})
        attempts += 1
    random.shuffle(pairs)
    return pairs


# ── Stats & display ────────────────────────────────────────────────────────────

def print_stats(pairs: list[dict]) -> None:
    total = len(pairs)
    avg_in  = sum(len(p["input"])  for p in pairs) / total
    avg_out = sum(len(p["output"]) for p in pairs) / total

    print(f"\n{'='*62}")
    print(f"  Dataset Stats")
    print(f"{'='*62}")
    print(f"  Total pairs          : {total:,}")
    print(f"  Avg input length     : {avg_in:.1f} chars")
    print(f"  Avg output length    : {avg_out:.1f} chars")
    print(f"  Avg dirty overhead   : {avg_in - avg_out:+.1f} chars")
    print(f"  Base sentence pool   : {len(ALL_SENTENCES):,}")

    print(f"\n{'='*62}")
    print(f"  10 Random Examples")
    print(f"{'='*62}")
    samples = random.sample(pairs, min(10, total))
    for i, pair in enumerate(samples, 1):
        print(f"\n[{i}]")
        print(f"  INPUT : {pair['input']}")
        print(f"  OUTPUT: {pair['output']}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Generating {TARGET_PAIRS:,} (dirty, clean) training pairs ...")
    pairs = generate_pairs(TARGET_PAIRS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs):,} pairs -> {OUTPUT_PATH}")
    print_stats(pairs)


if __name__ == "__main__":
    main()
