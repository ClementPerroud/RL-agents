class dispatch:
    def __init__(self, *operations : list[str] | tuple[str]):
        self.operations = operations
        self.owner = None
    
    def __call__(self, func):
        return dispatch.wrapper(func, operations=self.operations)
    
    class wrapper:
        def __init__(self, fn, operations):
            self.fn = fn
            self.operations = operations

        def __set_name__(self, owner, name):
            for op in self.operations:
                owner.DISPATCH_MAPPING[op] = self.fn
        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)
        

