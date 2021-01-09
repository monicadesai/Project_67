# -*- coding: utf-8 -*-
"""
Data Science Project
"""

import glassdoor_scraper as gs 
import pandas as pd 

#path = "C:/Users/Documents/ds_salary_proj/chromedriver"

#path = "C:/Users/Monica/Desktop/Projects/Python Projects 1/allin1/Project_3_WebScraping+DataCollection+DataCleaning+EDAwithVisualization+ModelBuilding(End_to_End_Project)/chromedriver"

path = "C:/Users/Monica/Desktop/salary_prediction_app/chromedriver.exe"

df = gs.get_jobs('data scientist',1000, False, path, 15)

df.to_csv('glassdoor_jobs.csv', index = False)

