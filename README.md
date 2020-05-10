BDA-analytics-challenge
==============================

## Group Members: 
- Forename: Random
- Surname: Dude
- Matriculation Number: XXXXXX

- Forename: Another
- Surname: Dude
- Matriculation Number: XXXXXX

## Special Stuff
Explain special stuff here (e.g., things an evaluator has to know to evaluate your submission)


## Project Organization
------------
```
	├── README.md 							<-- this file. insert group members here
	├── __init__.py 						<-- makes this file a packages, requrired for test.py
	├── .gitignore 						    <-- prevents you from submitting several clutter files
	├── data
	│   ├── modeling
	│   │   ├── dev 						<-- your development set goes here
	│   │   ├── test 						<-- your test set goes here
	│   │   │   └── test.csv 				<-- dummy test file to get test passing
	│   │   └── train 						<-- your train set goes here
	│   ├── preprocessed 					<-- your preprocessed data
	│   └── raw								<-- the given raw data for modeling
	│       ├── actors.csv
	│       ├── countries.csv
	│       ├── directors.csv
	│       ├── genres.csv
	│       ├── locations.csv
	│       ├── movie_tags.csv
	│       ├── movies.csv
	│       ├── ratings.csv
	│       └── tags.csv
	├── docs								<-- explanation of raw input data
	│   └── doc_train_data.pdf
	├── models								<-- dump models here
	├── notebooks							<-- your playground for juptyer notebooks
	├── requirements.txt 					<-- required packages to run your submission (mandatory for submission)
	├── src
	│   ├── additional_features.py 			<-- your creation of additional features goes here
	│   ├── predict.py 						<-- use your model to make predictions on final test set (mandatory for submission)
	│   ├── preprocessing.py 				<-- your preprocessing script goes here
	│   └── train.py 						<-- your development set goes here (mandatory for submission)
	└── test.py 							<-- simple sanity unittest to check your submission
```