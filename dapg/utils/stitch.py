import pickle
import os
import sys

demo_path = sys.argv[1]
output_path = sys.argv[2]

#filenames = ['demo_Oct_28_20_47/d_Oct_28_20_47.pickle','demo_Oct_28_20_54/d_Oct_28_20_54.pickle',
# 'demo_Oct_28_20_53/d_Oct_28_20_53.pickle', 'demo_Oct_28_20_57/d_Oct_28_20_57.pickle']

dirnames = os.listdir(demo_path)
all_demos = []
for d in dirnames:
    demo_dir = os.path.join(demo_path, d)
    for c in os.listdir(demo_dir):
        if(os.path.splitext(c)[1] == '.pickle'):
            fname = os.path.join(demo_dir,c)
            demo = pickle.load(open(fname,'rb'))
            all_demos.append(demo)

with open(output_path,'wb') as outfile:
    pickle.dump(all_demos,outfile)
