from unittest import TestCase

from expresso.pycas import *
import numpy as np

i,j = symbols('i,j',type=Types.Integer)
x,y = symbols("x, y",type=Types.Real)
z = Symbol('z',type=Types.Complex)

f = Function('f',argc = 1)
g = Function('g',argc = 2)

class TestAlgebra(TestCase):

    def test_factor(self):
        expr = ((-2*x**(z)*(y+2*y/6)/x**(z-1)-y*(x+x/3)+42*3*x**2/(3*x*y**((x-2*x)/x)))+4*x*y)/(x*y)
        self.assertEqual(expr.evaluate(), S(42))

    def test_derivative(self):
        expr = derivative(derivative(log(exp(x**y)*x**2),x),y)
        self.assertEqual(expr.evaluate(),x**(y-1)*(y*log(x)+1))


class TestCompilers(TestCase):

    def assertNumpyEqual(self,a,b):
        return self.assertSequenceEqual(list(a),list(b))

    def test_numpy_compiler(self):
        abs_expr = numpyfy(piecewise((x,x>=0),(-x,True)))
        self.assertNumpyEqual(abs_expr(x=[-np.pi,3,-2.5,0,np.e,np.pi,-42]),[np.pi,3,2.5,0,np.e,np.pi,42])

        random_numbers = np.random.rand(100)
        array_expr = numpyfy(array('rand',random_numbers)( floor(array('rand',random_numbers)(i)*100)),restype=float)

        self.assertNumpyEqual( array_expr(i = range(10)), random_numbers[(random_numbers[range(10)]*100).astype(int)] )

    def test_compiler_consistency(self):

        def my_function_python(x,y):
            return (abs(x)/(1+abs(y)))%1

        my_function_ccode = '''
        double myfunction(double x,double y){
            return fmod(abs(x)/(1+abs(y)),1);
        }
        '''

        pycas_custom_function = custom_function("myfunction",
                                        argc = 2,
                                        python_function=my_function_python,
                                        ccode = my_function_ccode,
                                        return_type=Types.Real)

        npx,npy = np.meshgrid(np.linspace(-10, 10, 1001),np.linspace(-9.5, 9.5, 950))

        snpx = array('np_x',npx+2*np.random.rand(*npy.shape))
        p = parameter('p',1)

        expr = piecewise((pi*snpx(sin(x)*cos(y)*1000,y*50),y>0),(e*pycas_custom_function(y-p*x,x+p*y)*10,True))

        fs_def = FunctionDefinition('f_single',(x,y),expr,return_type=Types.Real,parallel=False)
        fp_def = FunctionDefinition('f_parallel',(x,y),expr,return_type=Types.Real,parallel=True)

        clib = ccompile(fp_def,fs_def)
        nlib = ncompile(fp_def,fs_def)

        p.set_value(2)

        lf = lambdify(expr)
        N = mpmathify(expr)

        for (vx,vy) in zip(10*(np.random.rand(1000)-0.5),10*(np.random.rand(1000)-0.5)):
            self.assertTrue( np.isclose(nlib.f_single(vx,vy),lf(x=vx,y=vy)) )
            self.assertTrue( np.isclose(lf(x=vx,y=vy),float(N(x=vx,y=vy))) )

        lib_f_single_t = clib.f_single(npx,npy)
        lib_f_parallel_t = clib.f_parallel(npx,npy)
        cf_single_t = nlib.f_single(npx,npy)
        cf_parallel_t = nlib.f_parallel(npx,npy)

        self.assertTrue( np.allclose(lib_f_single_t, lib_f_parallel_t) )
        self.assertTrue( np.allclose(cf_single_t, cf_parallel_t) )
        self.assertTrue( np.allclose(cf_parallel_t, lib_f_single_t) )
