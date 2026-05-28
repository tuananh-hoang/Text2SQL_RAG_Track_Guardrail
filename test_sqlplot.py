from sqlglot import parse_one, exp
sql = "SELECT product, revenue FROM product_sales WHERE region = 'North'"

tree = parse_one(sql, dialect="sqlite")

if isinstance(tree, exp.Select):
    print("Đây là SELECT")
else:
    print("Không phải SELECT")