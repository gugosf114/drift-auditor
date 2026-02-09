import json

with open(r'C:\Users\georg\Desktop\Claude_Context\conversations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Type: {type(data).__name__}")

if isinstance(data, list):
    print(f"Total conversations: {len(data)}")
    if data:
        print(f"First conversation keys: {list(data[0].keys())}")
        c = data[0]
        print(f"Title: {c.get('title', '?')}")
        print(f"Created: {c.get('create_time', '?')}")
        
        # ChatGPT format uses nested mapping structure
        mapping = c.get('mapping', {})
        print(f"Mapping entries: {len(mapping)}")
        
        # Count messages
        msg_count = 0
        for node_id, node in mapping.items():
            msg = node.get('message')
            if msg and msg.get('content', {}).get('parts'):
                msg_count += 1
        print(f"Messages in first conv: {msg_count}")
        
        if mapping:
            first_node = list(mapping.values())[1]
            msg = first_node.get('message', {})
            if msg:
                print(f"Message keys: {list(msg.keys())}")
                role = msg.get('author', {}).get('role', '?')
                parts = msg.get('content', {}).get('parts', [])
                print(f"Role: {role}")
                if parts:
                    print(f"First part (100 chars): {str(parts[0])[:100]}")
    
    # Stats
    total_msgs = 0
    conv_sizes = []
    for conv in data:
        mapping = conv.get('mapping', {})
        n = sum(1 for node in mapping.values() 
                if node.get('message') and node['message'].get('content', {}).get('parts'))
        total_msgs += n
        title = conv.get('title', '?')
        conv_sizes.append((n, title[:60] if title else '?'))
    
    print(f"\nTotal messages across all conversations: {total_msgs}")
    print(f"Average messages per conversation: {total_msgs / max(len(data),1):.1f}")
    
    conv_sizes.sort(reverse=True)
    print(f"\nTop 10 longest conversations:")
    for n, name in conv_sizes[:10]:
        print(f"  {n:4d} msgs: {name}")
