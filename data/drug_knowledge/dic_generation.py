import pickle
import re
from collections import defaultdict

import pandas as pd
from unidecode import unidecode


def check_combination(libelle):
    words = libelle.split()
    for i in range(len(words) - 1):
        if re.search(r"[A-Z]+\/[A-Z]+", f"{words[i]} {words[i+1]}"):
            return True
    return False


def run():
    atc_classif = pd.read_excel("atc_2023.xls")
    atc_classif = atc_classif[["ATC_code", "Libellé français"]]
    atc_classif["Libellé français"] = atc_classif["Libellé français"].astype(str)
    atc_classif["to_dic"] = atc_classif["Libellé français"].apply(
        lambda x: unidecode(x.lower())
    )
    atc_classif = atc_classif.rename(columns={"ATC_code": "ATC", "to_dic": "STRING"})

    ruim = pd.read_csv("CIS.csv")
    ruim = ruim[["codeATC", "libelle"]]
    ruim["libelle"] = ruim["libelle"].astype(str)
    ruim["combination"] = False
    ruim["to_dic"] = ruim["libelle"].apply(lambda x: unidecode(x.split()[0].lower()))
    ruim["combination"] = ruim["libelle"].apply(check_combination)
    ruim = ruim.rename(columns={"codeATC": "ATC", "to_dic": "STRING"})

    umls = pd.read_pickle("../umls/atc_str_2023AB.pkl")
    umls = umls[["ATC", "STR"]]
    umls["STR"] = umls["STR"].astype(str)
    umls["to_dic"] = umls["STR"].apply(lambda x: unidecode(x.lower()))
    umls = umls.drop_duplicates(keep=False)
    umls = umls.rename(columns={"to_dic": "STRING"})

    # output is a dict of set

    drug_dic = defaultdict(set)

    for count, line in atc_classif.iterrows():
        drug_dic[line["ATC"]].add(line["STRING"])

    for count, line in ruim.iterrows():
        drug_dic[line["ATC"]].add(line["STRING"])

    for count, line in umls.iterrows():
        drug_dic[line["ATC"]].add(line["STRING"])

    # #save to pickle
    with open("final_dict.pkl", "wb") as f:
        pickle.dump(drug_dic, f)

    print("file saved as final_dict.pkl")


if __name__ == "__main__":
    run()
