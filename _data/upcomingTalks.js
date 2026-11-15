// Upcoming talks, shown on the landing page and at the top of /talks/.
// Only talks on or after today's date (US Pacific) are rendered, so a talk
// drops off by itself once it has happened; the daily scheduled build in
// .github/workflows/pages.yml keeps the live site current. Afterwards, move
// the talk to "Past Talks" in pages/talks.md.
//
// Fields: date ("YYYY-MM-DD"), endDate (optional, for multi-day talks),
// event, url (optional), title (optional), location, image (optional).
const talks = [
  {
    date: "2026-11-17",
    event: "MLOps World GenAI Summit 2026",
    url: "https://mlopsworld.com/",
    title: "Exit Strategy: A 12-Step Playbook for Moving from Closed to Open-Source AI",
    location: "Austin, TX",
    image: "/images/genai-summit-2026.png",
  },
];

const today = new Intl.DateTimeFormat("en-CA", {
  timeZone: "America/Los_Angeles",
}).format(new Date()); // YYYY-MM-DD

module.exports = talks
  .filter((t) => (t.endDate || t.date) >= today)
  .sort((a, b) => a.date.localeCompare(b.date));
