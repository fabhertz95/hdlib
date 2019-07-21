import os
import platform

import cffi

if __name__ == "__main__":
    ffi = cffi.FFI()
    ffi.cdef('''
    int test1();
    void test2(int * const var);
    ''')

    c_lib = ffi.dlopen(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_%s.so' % (platform.machine())
    ))

    def test1():
        return c_lib.test1()

    def test2(var):
        c_lib.test2(var)
                
        return var

    var1 = test1()
    var2 = ffi.new("int *", 0)
    test2(var2)
    print(var1)
    print(var2[0])
    
    ffi.dlclose(c_lib)

