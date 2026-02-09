import json

with open(r'C:\Users\georg\Desktop\Claude_Context\Data Export 2 8 26\conversations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Type: {type(data).__name__}")

if isinstance(data, list):
    print(f"Total conversations: {len(data)}")
    if data:
        print(f"First conversation keys: {list(data[0].keys())}")
        c = data[0]
        print(f"Name: {c.get('name', '?')}")
        print(f"Created: {c.get('created_at', '?')}")
        msgs = c.get('chat_messages', [])
        print(f"Messages in first conv: {len(msgs)}")
        if msgs:
            print(f"Message keys: {list(msgs[0].keys())}")
            print(f"First msg sender: {msgs[0].get('sender', '?')}")
            txt = str(msgs[0].get('text', ''))[:100]
            print(f"First msg text: {txt}")
    
    # Stats
    total_msgs = 0
    conv_sizes = []
    for conv in data:
        n = len(conv.get('chat_messages', []))
        total_msgs += n
        conv_sizes.append((n, conv.get('name', '?')[:60]))
    
    print(f"\nTotal messages across all conversations: {total_msgs}")
    print(f"Average messages per conversation: {total_msgs / len(data):.1f}")
    
    conv_sizes.sort(reverse=True)
    print(f"\nTop 10 longest conversations:")
    for n, name in conv_sizes[:10]:
        print(f"  {n:4d} msgs: {name}")

elif isinstance(data, dict):
    print(f"Top-level keys: {list(data.keys())}")
