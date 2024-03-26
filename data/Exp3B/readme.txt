You have already done zero-shot counter speech generation, where no gold label from actual annoated datasets have been used. Now we will use the existing finetuned saved model and do a second stage finetuning using some gold annotated hate-counter speech pais.

This experiment checks even if we are able to generate counterspeeches using synthetic dataset. Further finetuning the model will lead to performace improvement of the generation models.

This setting is called few-shot setting.

i) few-shot setting:
	For Bengali:
	Randomly select 100, 200 train data points from the actual gold trainset and use them to perform 2nd stage finetuning on the saved generation model(which was trained with synthetic Bengali data.)
        (train on: ->100/ 200 datapoints from the gold train data, validation: the actual gold validation set of the Bengali dataset, test: the actual gold testset of the Bengali dataset)
		EnglishToBengali training-> (bengali_train_pairsNew.csv). test(bengali_test_pairsNew.csv), validation(bengali_val_pairsNew.csv) 
		HindiToBengali training-> (bengali_train_pairsNew.csv). test(bengali_test_pairsNew.csv), validation(bengali_val_pairsNew.csv) 


	For Hindi:
	Randomly select 100, 200 train data points from the actual gold trainset and use them to perform 2nd stage finetuning on the saved generation model(which was trained with synthetic Hindi data.)
        (train on: ->100/ 200 datapoints from the gold train data, validation: the actual gold validation set of the Hindi dataset, test: the actual gold testset of the Hindi dataset)
		EnglishToHindi training->(hindi_train_pairsNew.csv). test(hindi_test_pairsNew.csv), validation(hindi_val_pairsNew.csv)
		BengaliToHindi training->(hindi_train_pairsNew.csv). test(hindi_test_pairsNew.csv), validation(hindi_val_pairsNew.csv)
