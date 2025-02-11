# 创建一个示例的物品数据库，每个物品都有编号和分类标签
items = [
    {'id': 1, 'tags': ['electronics', 'gadgets', 'smartphone']},
    {'id': 2, 'tags': ['clothing', 'fashion', 'jacket']},
    {'id': 3, 'tags': ['electronics', 'camera', 'digital']},
    {'id': 4, 'tags': ['books', 'fiction', 'mystery']},
    {'id': 5, 'tags': ['sports', 'equipment', 'tennis']}
]


# 定义一个函数，用于执行联合查询
def search_items(database, **kwargs):
    results = []

    for item in database:
        match_all = True

        for key, value in kwargs.items():
            if key == 'tag':
                # 如果关键字是'tag'，则检查物品的分类标签中是否包含值
                if not any(value in tag for tag in item['tags']):
                    match_all = False
                    break
            elif key == 'id':
                # 如果关键字是'id'，则检查物品的编号是否匹配值
                if item['id'] != value:
                    match_all = False
                    break
            else:
                # 其他关键字不支持
                print(f"不支持的关键字: {key}")

        if match_all:
            results.append(item)

    return results


# 示例查询
query_result = search_items(items, tag='electronics')

# 打印查询结果
if query_result:
    for item in query_result:
        print(f"物品编号: {item['id']}, 分类标签: {item['tags']}")
else:
    print("没有找到匹配的结果。")
