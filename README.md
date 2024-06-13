# Football Data Analysis via Pass Map embedding by GCN
## 0. Files
### matches folder
All the open match data that statsbomb provides. Refer to <https://github.com/statsbomb/open-data/tree/master/data>. 'matches.zip' is just a zip file of this matches data.

### Team33_code.ipynb
Our code for analyzing the data. It has two parts largely which is player-level analysis and team-level analysis. For player-level analysis, we applied GCN for embedding the graph, and for team-level analysis, we applied pooling the embedded data and perform neural networks for analysis. High level ideas can be found in other submit file, research_20220648_DeukhwanCho_poster.pdf (which is not a file in research_20220648_DeukhwanCho_poster.zip)

### sb_passmaps.ipynb 
Reference for our generate_pass_map(). It does not require other files as dependency, and can run soley. The file is slightly changed, fitting to our cases. Also the file supports visualizations of the pass map so finding match_id in Team33_code.ipynb and putting it to sb_passmaps.ipynb MATCH_ID variable can generate visualized pass maps for each team in that match.
'''
MATCH_ID = 3869685 # Put match id here
df = sb.events(MATCH_ID)
df.head()
'''
Refer to <https://sharmaabhishekk.github.io/projects/passmap>

### pickle file
Some code in the Team33_code.ipynb has heavy computational burden. So we provided pickle file for each results.

## 1. File upload
Team33_code.ipynb file assumes that the user is using google colab environment.
'''
from google.colab import drive
drive.mount('/content/drive')

os.chdir('/content/drive/MyDrive/CS471 Project') # User path
'''
Mount the drive to the environment and change the user path via os.chdir function. Above is just an example. 

### matches folder
To make match_list array from folder 'matches', we access its path as './matches/'. Assure that the folder is in the same location as working directory. 

### pickle file
If you want to save the results after playel_level GCN, run this code(which is already implemented in 파일명.ipynb).
'''
with open('player_style.pickle', 'wb') as f:
  pickle.dump(player_style, f)
'''
If you want to just run the code using given pickle file and not run GCN, use the code below(which is already implemented in Team33_code.ipynb).
'''
with open('player_style.pickle', 'rb') as f:
  player_style = pickle.load(f)
'''

## 2. Dependencies
## statsbombpy
'''
Name: statsbombpy
Version: 1.13.0
Summary: easily stream StatsBomb data into Python
Home-page: https://github.com/statsbomb/statsbombpy
Author: StatsBomb
Author-email: support@statsbombservices.com
'''

## scipy
'''
Name: scipy
Version: 1.11.4
Summary: Fundamental algorithms for scientific computing in Python
Home-page: https://scipy.org/
Author: 
Author-email: 
'''

## torch_geometric
'''
Name: torch_geometric
Version: 2.5.3
Summary: Graph Neural Network Library for PyTorch
Home-page: 
Author: 
Author-email: Matthias Fey <matthias@pyg.org>
'''