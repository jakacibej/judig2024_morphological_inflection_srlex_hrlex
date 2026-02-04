import pickle
from collections import defaultdict as dd

class PatternPredictorAndGenerator:

    def __init__(self):
        """Load all the external resources (lists of ending parts, models and patterns)"""
        self.Ncm_model = pickle.load(open("./hrLex_models/model_hrLex_LogisticRegression_Ncm.sav", "rb"))
        self.Npm_model = pickle.load(open("./hrLex_models/model_hrLex_LogisticRegression_Npm.sav", "rb"))
        self.Ncf_model = pickle.load(open("./hrLex_models/model_hrLex_LogisticRegression_Ncf.sav", "rb"))
        self.Npf_model = pickle.load(open("./hrLex_models/model_hrLex_LogisticRegression_Npf.sav", "rb"))
        self.Ncn_model = pickle.load(open("./hrLex_models/model_hrLex_LogisticRegression_Ncn.sav", "rb"))
        self.Npn_model = pickle.load(open("./hrLex_models/model_hrLex_LogisticRegression_Npn.sav", "rb"))
        self.Vm_model = pickle.load(open("./hrLex_models/model_hrLex_LogisticRegression_Vm.sav", "rb"))
        self.Ag_model = pickle.load(open("./hrLex_models/model_hrLex_LogisticRegression_Ag.sav", "rb"))
        self.Ap_model = pickle.load(open("./hrLex_models/model_hrLex_LogisticRegression_Ap.sav", "rb"))
        self.As_model = pickle.load(open("./hrLex_models/model_hrLex_LogisticRegression_As.sav", "rb"))
        self.Rg_model = pickle.load(open("./hrLex_models/model_hrLex_LogisticRegression_Rg.sav", "rb"))
        self.Rr_model = pickle.load(open("./hrLex_models/model_hrLex_LogisticRegression_Rr.sav", "rb"))

        self.list_of_endings_for_Ncm = [line.strip("\n").split("\t")[0] for line in open("./ending_parts/hrLex_Ncm_endings.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_endings_for_Npm = [line.strip("\n").split("\t")[0] for line in open("./ending_parts/hrLex_Npm_endings.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_endings_for_Ncf = [line.strip("\n").split("\t")[0] for line in open("./ending_parts/hrLex_Ncf_endings.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_endings_for_Npf = [line.strip("\n").split("\t")[0] for line in open("./ending_parts/hrLex_Npf_endings.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_endings_for_Ncn = [line.strip("\n").split("\t")[0] for line in open("./ending_parts/hrLex_Ncn_endings.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_endings_for_Npn = [line.strip("\n").split("\t")[0] for line in open("./ending_parts/hrLex_Npn_endings.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_endings_for_Vm = [line.strip("\n").split("\t")[0] for line in open("./ending_parts/hrLex_Vm_endings.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_endings_for_Ag = [line.strip("\n").split("\t")[0] for line in open("./ending_parts/hrLex_Ag_endings.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_endings_for_Ap = [line.strip("\n").split("\t")[0] for line in open("./ending_parts/hrLex_Ap_endings.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_endings_for_As = [line.strip("\n").split("\t")[0] for line in open("./ending_parts/hrLex_As_endings.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_endings_for_Rg = [line.strip("\n").split("\t")[0] for line in open("./ending_parts/hrLex_Rg_endings.tsv", "r", encoding="UTF-8").readlines()]
        self.list_of_endings_for_Rr = [line.strip("\n").split("\t")[0] for line in open("./ending_parts/hrLex_Rr_endings.tsv", "r", encoding="UTF-8").readlines()]

        self.dict_of_fpos_and_lists = {"Ncm": self.list_of_endings_for_Ncm,
                                  "Npm": self.list_of_endings_for_Npm,
                                  "Ncf": self.list_of_endings_for_Ncf,
                                  "Npf": self.list_of_endings_for_Npf,
                                  "Ncn": self.list_of_endings_for_Ncn,
                                  "Npn": self.list_of_endings_for_Npn,
                                  "Vm": self.list_of_endings_for_Vm,
                                  "Ag": self.list_of_endings_for_Ag,
                                  "Ap": self.list_of_endings_for_Ap,
                                  "As": self.list_of_endings_for_As,
                                  "Rg": self.list_of_endings_for_Rg,
                                  "Rr": self.list_of_endings_for_Rr}

        self.dict_of_fpos_and_models = {"Ncm": self.Ncm_model,
                                  "Npm": self.Npm_model,
                                  "Ncf": self.Ncf_model,
                                  "Npf": self.Npf_model,
                                  "Ncn": self.Ncn_model,
                                  "Npn": self.Npn_model,
                                  "Vm": self.Vm_model,
                                  "Ag": self.Ag_model,
                                  "Ap": self.Ap_model,
                                  "As": self.As_model,
                                  "Rg": self.Rg_model,
                                  "Rr": self.Rr_model}


        # POPULATE DICTIONARY WITH MORPHOLOGICAL PATTERNS
        self.dict_morphological_patterns = dd()
        for line in open("./patterns/hrLex_morphological_patterns.tsv", "r", encoding="UTF-8").readlines()[1:]:  # SKIP HEADER
            pattern_code = line.strip("\n").split("\t")[0]
            pattern = line.strip("\n").split("\t")[2]
            self.dict_morphological_patterns[pattern_code] = pattern

        # LIST OF FUNDAMENTAL MSDS USED FOR LEMMA FORMS
        # CAUTION - DO **NOT** CHANGE THE ORDER OF MSDs IN THE FILE UNDER ANY CIRCUMSTANCES.
        self.list_of_msds_for_lemmas = [line.strip("\n") for line in open("./resources/list_of_msds_for_lemmas.tsv", "r", encoding="UTF-8").readlines()]


    def vectorize_lemma(self, lemma, fpos):
        try:
            list_of_endings = self.dict_of_fpos_and_lists[fpos]
        except:
            raise Exception("Invalid Multext East v6 MSD.")

        list_to_vectorize_lemma = []
        for ending in list_of_endings:
            if lemma.lower().endswith(ending.lower()):
                list_to_vectorize_lemma.append(1)
            else:
                list_to_vectorize_lemma.append(0)

        return list_of_endings, list_to_vectorize_lemma


    def predict_morphological_pattern(self, lemma, fpos):

        # VECTORIZE LEMMA
        list_of_endings, vectorized_lemma = self.vectorize_lemma(lemma, fpos)

        # SELECT RELEVANT MODEL
        relevant_model = self.dict_of_fpos_and_models[fpos]

        # PREDICT MORPHOLOGICAL PATTERN CODE
        predicted_pattern_code = relevant_model.predict([vectorized_lemma])

        return predicted_pattern_code[0]

    def generate_forms_based_on_pattern(self, lemma, pattern_code):
        """FUNCTION - GENERATE FORMS FROM PATTERN, LEMMA AND POS"""
        # CHECK IF THE PATTERN CODE IS AVAILABLE IN THE PATTERN DICTIONARY, OTHERWISE
        if pattern_code in self.dict_morphological_patterns:
            self.pattern = self.dict_morphological_patterns[pattern_code]
        else:
            return None

        # POPULATE PATTERN DICTIONARY
        self.pattern_dictionary = dd(list)
        for element in self.pattern.split(", "):
            self.tag, self.ending_part = element.split(": ")
            for ending in self.ending_part.split("|"):
                self.pattern_dictionary[self.tag].append(ending.strip("~").replace("Ø", ""))

        # DETERMINE THE BASIC MSD OF THE LEMMA (i.e. THE FIRST AVAILABLE ON THE PRIORITY LIST)
        for item in self.list_of_msds_for_lemmas:
            if item in self.pattern_dictionary:
                self.basic_msd = item
                break

        # GET IMMUTABLE PART OF THE LEMMA
        ending_part_for_basic_msd = self.pattern_dictionary[self.basic_msd][0]
        if not ending_part_for_basic_msd == "":
            self.immutable_part = lemma[:(len(lemma) - len(ending_part_for_basic_msd))]
        else:
            self.immutable_part = lemma

        # POPULATE DICTIONARY WITH FORMS
        self.dict_with_forms = dd(list)
        for key in self.pattern_dictionary:
            for ending_part in self.pattern_dictionary[key]:
                self.dict_with_forms[key].append(f"{self.immutable_part}{ending_part}")

        return self.dict_with_forms



# INSTANCE OF PATTERN PREDICTOR AND GENERATOR
pattern_predictor_and_generator = PatternPredictorAndGenerator()


test_lemma = "stol"
test_fpos = "Ncm"

pattern_code = pattern_predictor_and_generator.predict_morphological_pattern(lemma=test_lemma, fpos=test_fpos)
print(pattern_code)

generated_forms = pattern_predictor_and_generator.generate_forms_based_on_pattern(lemma=test_lemma, pattern_code=pattern_code)
print(generated_forms)