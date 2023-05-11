import pandas

table = pandas.read_table("electrode-locations.ced", usecols=[""])
table.drop(labels=["labels"])
# table.to_csv("electrode-locations.dat", sep=" ")
print(table)
