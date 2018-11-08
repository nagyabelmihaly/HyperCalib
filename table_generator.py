import csv
import os

from rmsae import RMSAE
from rmsre import RMSRE
from weighted_error import WeightedError
from neo_hooke import NeoHooke
from ogden import Ogden
from mooney_rivlin import MooneyRivlin
from yeoh import Yeoh
from arruda_boyce import ArrudaBoyce

models = [NeoHooke(), MooneyRivlin(), Yeoh(), Ogden(1), Ogden(2), Ogden(3), ArrudaBoyce()]
error_functions = [RMSAE, RMSRE]
weightdata = [([1, 0, 0], 'UT'),
              ([0, 1, 0], 'ET'),
              ([0, 0, 1], 'PS'),
              ([1, 1, 0], 'UT+ET'),
              ([0, 1, 1], 'ET+PS'),
              ([1, 0, 1], 'PS+UT'),
              ([1, 1, 1], 'UT+ET+PS')]

#models = [Ogden(1), ArrudaBoyce()]
#error_functions = [RMSAE]
#weightdata = [([1, 0, 0], 'UT'),
#              ([0, 1, 0], 'ET')]
with open(os.getcwd() + "/Table.txt", "w") as text_file:
    for exp in range(1, 4):
        for model in models:
            for error_function in error_functions:
                error_type = 'absz.' if error_function == RMSAE else 'rel.'
                for weights, weightname in weightdata:
                    filename = os.getcwd() + '/Reports/EXP{}-{}-{}-{}.csv'.format( \
                                    exp, weightname, model.name, error_function.shortname)
                    with open(filename, 'r', newline='') as f:
                        lines = f.readlines()
                    csvreader = csv.reader(lines, delimiter=';')
                    rowindex = -1
                    params = []
                    r2s = []
                    for row in csvreader:
                        rowindex += 1
                        if rowindex == 0:
                            continue
                        if rowindex == 1:
                            for col in row:
                                r2s.append(float(col))
                            continue
                        params.append(float(row[1]))
                    #if isinstance(model, Ogden):
                    #    n = len(params)//2
                    #    mu = params[:n]
                    #    alpha = params[n:]
                    #    mutext = ', '.join(['{:.3g}'.format(m) for m in mu])
                    #    alphatext = ', '.join(['{:.3g}'.format(a) for a in alpha])
                    #    paramtext = '\\mu=[{}], \\alpha=[{}]'.format(mutext, alphatext)
                    #else:
                    #    paramtext = ', '.join('{}={:.3g}'.format(pname, p) for pname, p in zip(model.paramnames_latex, params))
                    paramtext = ', '.join('{:.3g}'.format(p) for p in params)
                    r2text = '&'.join(['\\{}{{{:.5g}}}'.format('textbf' if weights[defmode] > 0 else 'textit', r2s[defmode]) for defmode in range(3)])
                    text_file.write('{}&{}&{}&[{}]&{}\n\\\\\n\hline\n'.format(exp, model.name, error_type, paramtext, r2text))