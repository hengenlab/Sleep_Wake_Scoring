import Sleep_Wake_Scoring as sw
import os.path as op
import os


fl_list = ['CAF52_1123.json',
           'CAF100.json',
           'CAF101.json',
           'CAF102.json',
           'CAF107.json',
           'CAF22.json',
           'CAF26.json',
           'CAF37_0828.json',
           'CAF37.json',
           'CAF40_0905.json',
           'CAF40_0909.json',
           'CAF42.json',
           'CAF48.json',
           'CAF49_0108.json',
           'CAF49_1030.json',
           'CAF49_1106.json',
           'CAF49.json',
           'CAF49_late.json',
           'CAF50.json',
           'CAF52_1021.json',
           'CAF52_1027.json',
           'CAF52_1030.json',
           'CAF52_1104.json',
           'CAF52_1111.json',
           'CAF52_1116.json',
           'CAF52_1128.json',
           'CAF52_1202.json',
           'CAF52_1205.json',
           'CAF52.json',
           'CAF60.json',
           'CAF61.json',
           'CAF69_0108.json',
           'CAF69_1229.json',
           'CAF69_1230.json',
           'CAF72.json',
           'CAF78_0312.json',
           'CAF78.json',
           'CAF81_0306.json',
           'CAF82.json',
           'CAF84_0402.json',
           'CAF84.json',
           'CAF88.json',
           'CAF89_0416.json',
           'CAF89.json',
           'CAF90.json',
           'CAF95.json',
           'EAB42.json',
           'KDR14_1019.json',
           'KDR14_1027.json',
           'KDR14.json',
           'KDR27_0106.json',
           'KDR27_0114.json',
           'KDR27.json',
           'KDR36.json',
           'KDR48.json']

# data = {'CAF52_1123.json' : 2:10,
#         'CAF100.json': 2:10,
#         'CAF101.json': 2:10,
#         'CAF102.json': 2:10,
#         'CAF107.json': 2:10,
#         'CAF22.json': 2:10,
#         'CAF26.json': 2:10,
#         'CAF37_0828.json': 2:10,
#         'CAF37.json': 2:10,
#         'CAF40_0905.json': 2:10,
#         'CAF40_0909.json': 2:10,
#         'CAF42.json': 2:10,
#         'CAF48.json': 2:10,
#         'CAF49_0108.json': 2:10,
#         'CAF49_1030.json': 2:10,
#         'CAF49_1106.json': 2:10,
#         'CAF49.json': 2:10,
#         'CAF49_late.json': 2:10,
#         'CAF50.json': 2:10,
#         'CAF52_1021.json': 2:10,
#         'CAF52_1027.json': 2:10,
#         'CAF52_1030.json': 2:10,
#         'CAF52_1104.json': 2:10,
#         'CAF52_1111.json': 2:10,
#         'CAF52_1116.json': 2:10,
#         'CAF52_1128.json': 2:10,
#         'CAF52_1202.json': 2:10,
#         'CAF52_1205.json': 2:10,
#         'CAF52.json': 2:10,
#         'CAF60.json': 2:10,
#         'CAF61.json': 2:10,
#         'CAF69_0108.json': 2:10,
#         'CAF69_1229.json': 2:10,
#         'CAF69_1230.json': 2:10,
#         'CAF72.json': 2:10,
#         'CAF78_0312.json': 2:10,
#         'CAF78.json': 2:10,
#         'CAF81_0306.json': 2:10,
#         'CAF82.json': 2:10,
#         'CAF84_0402.json': 2:10,
#         'CAF84.json': 2:10,
#         'CAF88.json': 2:10,
#         'CAF89_0416.json': 2:10,
#         'CAF89.json': 2:10,
#         'CAF90.json': 2:10,
#         'CAF95.json': 2:10,
#         'EAB42.json': 2:10,
#         'KDR14_1019.json': 2:10,
#         'KDR14_1027.json': 2:10,
#         'KDR14.json': 2:10,
#         'KDR27_0106.json': 2:10,
#         'KDR27_0114.json': 2:10,
#         'KDR27.json': 2:10,
#         'KDR36.json': 2:10,
#         'KDR48.json': 2:10]
#        }
dir_path = '/media/HlabShare/ckbn/train_sleep_wake_model/json_files/'

fileidx = 0
for fileidx,  fl in enumerate(fl_list):
    try:
        for i in range(2, 11, 1):
            print(f'Hour {i}')
            sw.load_data_for_sw_v2(op.join(dir_path, fl_list[fileidx]),
                                   hr=str(i))

        final_dir = '/hlabhome/kiranbn/git/Sleep_Wake_Scoring_p/'
        os.chdir(final_dir)
        final_file = 'data_tosave.csv'
        os.rename(final_file, 'data_tosave' +
                  op.splitext(fl_list[fileidx])[0] + '.csv')
    except Exception as e:
        print(e)
        with open (op.join(final_dir, 'error.txt'), 'a') as f:
            f.write(str(fileidx))
            f.write(str(fl))
            f.write(str(e))
            f.write("\n\n\n")
