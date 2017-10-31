def modify(dataset):
	temp[0]=1;
	target=["Gold"]
	for i in range(1,len(target)):
		if target[i]-target[i-1] >0:
			temp[i]=1;
		else:
			temp[i]=0;
	for i in range(0,len(target)):
		dataset["Gold"].iloc[i]=temp[i];
	return dataset;
	
def fillna(df):
	df['Gold'].fillna((df['Gold'].mean()), inplace=True)