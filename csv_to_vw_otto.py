# -*- coding: UTF-8 -*-

########################################################
# __Author__: Triskelion <info@mlwave.com>             #
# Kaggle competition "Display Advertising Challenge":  #
# http://www.kaggle.com/c/criteo-display-ad-challenge/ #
# Credit: Zygmunt Zając <zygmunt@fastml.com>           #
########################################################

from datetime import datetime
from csv import DictReader

def csv_to_vw(loc_csv, loc_output, train=True):
  """
  Munges a CSV file (loc_csv) to a VW file (loc_output). Set "train"
  to False when munging a test set.
  TODO: Too slow for a daily cron job. Try optimize, Pandas or Go.
  """
  start = datetime.now()
  print("\nTurning %s into %s. Is_train_set? %s"%(loc_csv,loc_output,train))
  
  with open(loc_output,"wb") as outfile:
    for e, row in enumerate( DictReader(open(loc_csv)) ):
	
	  #Creating the features
      numerical_features = ""
      categorical_features = ""
      for k,v in row.items():
        if k not in ["target","id"]:
          if "f" in k: # numerical feature, example: I5
            if len(str(v)) > 0: #check for empty values
              numerical_features += " %s:%s" % (k,v)
          if "c" in k: # categorical feature, example: C2
            if len(str(v)) > 0:
              categorical_features += " %s" % v
			  
	  #Creating the labels		  
      if train: #we care about labels
        print row['target']
        print row['target'] == "Class_1"
        if row['target'] == "Class_1":
          label = 1
        if row['target'] == "Class_2":
          label = 2
        if row['target'] == "Class_3":
          label = 3
        if row['target'] == "Class_4":
          label = 4
        if row['target'] == "Class_5":
          label = 5
        if row['target'] == "Class_6":
          label = 6
        if row['target'] == "Class_7":
          label = 7
        if row['target'] == "Class_8":
          label = 8
        if row['target'] == "Class_9":
          label = 9
        #else:
          #label = -1 #we set negative label to -1
        outfile.write( "%s '%s |i%s |c%s\n" % (label,row['target'],numerical_features,categorical_features) )
		
      else: #we dont care about labels
        outfile.write( "1 '%s |i%s |c%s\n" % (row['id'],numerical_features,categorical_features) )
      
	  #Reporting progress
      if e % 1000000 == 0:
        print("%s\t%s"%(e, str(datetime.now() - start)))

  print("\n %s Task execution time:\n\t%s"%(e, str(datetime.now() - start)))

#csv_to_vw("d:\\Downloads\\train\\train.csv", "c:\\click.train.vw",train=True)
#csv_to_vw("d:\\Downloads\\test\\test.csv", "d:\\click.test.vw",train=False)

