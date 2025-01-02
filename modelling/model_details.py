model_list = ['Li-SD', 'Li-D']
protocol_list = ['Milnes', 'ramp', 'step', 'staircase']

param_names = {
    'Li-SD': ['Kmax', 'Ku', 'halfmax', 'n', 'Vhalf'],
    'Li-D': ['Kb', 'Ku']
}

drug_list = ['dofetilide', 'bepridil', 'terfenadine', 'cisapride',
             'verapamil', 'ranolazine', 'quinidine', 'sotalol',
             'chlorpromazine', 'ondansetron', 'diltiazem', 'mexiletine']
lit_drug_conc = {
    'dofetilide': [1, 3, 10, 30],
    'verapamil': [30, 100, 300, 1000],
    'bepridil': [10, 30, 100, 300],
    'terfenadine': [3, 10, 30, 100],
    'cisapride': [1, 10, 100, 300],
    'ranolazine': [1000, 1e4, 3e4, 1e5],
    'quinidine': [100, 300, 1000, 10000],
    'sotalol': [1e4, 3e4, 1e5, 3e5],
    'chlorpromazine': [100, 300, 1000, 3000],
    'ondansetron': [300, 1000, 3000, 1e4],
    'diltiazem': [3000, 1e4, 3e4, 1e5],
    'mexiletine': [1e4, 3e4, 1e5, 3e5]}

dimless_conc_range = (10**-8, 0.65)
