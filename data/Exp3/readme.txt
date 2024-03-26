This experiment checks if we did not have any hate-counter pair data for Bengali and Hindi and how beneficial it would be to use the existing dataset in other languages.
For example, hate-counter pairs datasets are present in English(CONAN), and we have developed datasets for both Bengali and Hindi.
Now we want to generate counter speeches for Hindi hate speech without having any hate counter pairs for Hindi. What can be the solution?
1) We can translate English languages hate-counter pairs to Hindi hate-counter pairs and directly use this model to generate Hindi counter speeches.
2) We can also translate Bengali hate-counter pairs to Hindi hate-counter pairs and use this model to generate Hindi counter speeches.
Our assumption is as Bengali is closer to Hindi. The model trained on synthetically generated Hindi pairs from the Bengali dataset should perform better. Similar experiments for Bengali counter-speech generation.

This setting is called zero-shot setting.

i) Zero-shot setting:
	For Bengali:
		EnglishToBengali (train on: ->conanEnglishToBengali_train_pairs.csv, validation: conanEnglishToBengali_val_pairs.csv, test: the actual gold testset of the Bengali dataset[bengali_test_pairsNew.csv])
		HindiToBengali (train on: ->counterHindiToBengali_train_pairs.csv, validation: counterHindiToBengali_val_pairs.csv, test: the actual gold testset of the Bengali dataset[bengali_test_pairsNew.csv])


	For Hindi:
		EnglishToHindi (train on: ->conanEnglishToHindi_train_pairs.csv, validation: conanEnglishToHindi_val_pairs.csv, test: the actual gold testset of the Hindi dataset[hindi_test_pairsNew.csv])
		BengaliToHindi (train on: ->counterBengaliToHindi_train_pairs.csv, validation: counterBengaliToHindi_val_pairs.csv, test: the actual gold testset of the Hindidataset[hindi_test_pairsNew.csv])



Exp 3A : English to Bengali & Hindi To Bengali

Exp 3B : English to Hindi & Bengali to Hindi