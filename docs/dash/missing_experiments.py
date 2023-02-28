import glob

list_of_existing_files = glob.glob("./docs/dash/csvs/*.csv")
core_list = [1, 2, 4, 8, 16, 32]
implementation_list = ["openblas", "mkl", "netlib", "blis"]
solver_list = ["svrg", "qning-svrg", "qning-miso", "qning-ista", "miso", "ista", "fista", "catalyst-svrg", "catalyst-miso", "catalyst-ista", "acc-svrg", "auto"]
dataset_list = [["covtype", "1.721E-10"], ["alpha", "2.000E-10"], ["real-sim", "1.383E-09"], ["epsilon", "4.000E-10"], ["ocr", "4.000E-11"], ["rcv1", "1.280E-10"], ["webspam", "4.000E-10"], ["kddb", "5.191E-12"], ["criteo", "2.181E-12"], ["ckn_mnist", "1.667E-09"], ["svhn", "placeholder"]]


list_of_expected_files= list()

for core in core_list:
    for implementation in implementation_list:
        for solver in solver_list:
            for dataset in dataset_list:
                string = str(core) + "_" + implementation + "_l2_" + dataset[1] + "_5_4_" + solver + "_1e-05_" + dataset[0] + ".csv"
                list_of_expected_files.append(string) 



output = [x for x in list_of_expected_files if not x in list_of_existing_files or list_of_existing_files.remove(x)]

print(output)
print(len(list_of_expected_files))
print(len(list_of_existing_files))