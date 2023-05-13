'''
Fit output voltage-power curve of three kinds of solar batteries.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from io import StringIO
from typing import TypeAlias, TypeVar

from numpy.typing import NDArray
from numpy.polynomial.polynomial import polyder, polyroots, polyval
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


Float: TypeAlias = np.float64
FloatArr: TypeAlias = NDArray[Float]
T = TypeVar('T', Float, FloatArr)


plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['mathtext.it'] = 'Latin Modern Math'
plt.rcParams['font.family'] = 'Latin Modern Math'


def process_csv(s: str) -> tuple[FloatArr, FloatArr]:
    '''
    Read CSV content, use columns of `v` (voltage) and `po` (power), then put
    them into two NumPy ndarrays of float64.
    '''

    with StringIO(s, newline='') as f:
        df = pd.read_csv(f)
        arr = df.values.astype(Float)[:, (0, 2)]
        v, po = arr[:, 0], arr[:, 1]
    return v, po


def fit(v: FloatArr, po: FloatArr, degree: int = 5) -> Pipeline:
    '''Fit the given voltage and current arrays.'''

    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(v.reshape(-1, 1), po.reshape(-1, 1))
    return model


def plot(v: FloatArr, po: FloatArr, model: Pipeline) -> None:
    '''
    Plot the original data and the fit curve. We assume that the voltage data is
    ascending.
    '''

    vmin: Float = v.min()
    vmax: Float = v.max()
    curve_x = np.linspace(vmin, vmax, 100)
    curve_y = model.predict(curve_x.reshape(-1, 1))
    plt.plot(curve_x, curve_y, linestyle='dashed',
             color='#549A76', label='Fit curve')
    plt.plot(v, po, linestyle='solid',
             color='#4271AB', label='Original data')
    plt.xlabel('$V$ (V)')
    plt.ylabel('$P$ (mV)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    for i, j in zip((1, 2, 3), [7] * 3):
        print(f'==> Process: {i}')
        with open(f'../../data/report-5/tbl_3_{i}.csv',
                  'r', encoding='utf-8') as f:
            v, po = fit.process_csv(f.read())
            print('v =', v, 'po =', po, sep='\n')
            model = fit.fit(v, po, degree=j)
            print('popt =', model.named_steps['linear'].coef_, sep='\n')
            coef = model.named_steps['linear'].coef_.tolist()[0]
            print('popt print =', ' & '.join(['%.2f' % i for i in coef]))
            coef_der = polyder(coef)
            print('popt der =', coef_der, sep='\n')
            coef_der_roots = polyroots(coef_der)
            print('popt der roots =', coef_der_roots, sep='\n')
            v_when_po_max = np.float64(coef_der_roots[-1])
            print('v when po max =', v_when_po_max)
            po_max = polyval(v_when_po_max, coef)  # unit: mW
            print('po max =', po_max)
            r_when_po_max = v_when_po_max ** 2 / po_max  # unit: k ohm
            print('r when po max =', r_when_po_max)
            score = model.score(v.reshape(-1, 1), po.reshape(-1, 1))
            print('error rate = %f%%' % ((1 - score) * 100))
            # fit.plot(v, po, model)
            print()
