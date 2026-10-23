
# crawl8: topics and comments posted since the 10-15 crawl, newest topic first
new = []
for js in sorted(keep, key=lambda j: j["topic"]["postDate"], reverse=True):
    tp = js["topic"]
    cs = [c for c in js.get("comments", []) if c["postDate"] >= "2026-10-15"]
    if tp["postDate"] >= "2026-10-15" or cs:
        new.append(f"\n######## TOPIC {tp['id']} | {tp['postDate'][:10]} | {tp['authorName']} | {tp['title']}")
        if tp["postDate"] >= "2026-10-15":
            new.append(plain(str(tp.get("content", "")))[:4000])
        for c in sorted(cs, key=lambda c: c["postDate"]):
            new.append(f"--- {c['authorName']} | {c['postDate'][:16]} | v{c['votes']}\n{plain(c.get('content', ''))}")
(D / "new_since_1015.txt").write_text("\n".join(new))
print("new_since_1015 lines:", len(new))
