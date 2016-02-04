
pc = __import__(__name__.split('.')[0])

def get_symbols_in(expr):

    symbols = set()
    for e in pc.postorder_traversal(expr):
        if e.is_symbol:
            symbols.add(e)
    return symbols

