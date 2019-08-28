import sys
import os
import time

dataname = [  "U_1851_1a2", "U_1851_2c19", "U_1851_2c9", "U_1851_2d6", "U_1851_3a4",
              "U_1915", "U_2358", "U_463213", "U_463215", "U_488912", "U_488915",
						  "U_488917", "U_488918", "U_492992", "U_504607", "U_624504", "U_651739",
							"U_651744", "U_652065"]

for data in dataname :
	command = "python qsar_classify.py all "+data
	print command
	os.system(command)
