import platform


DATASET2FILEID = {
    "Chickenpox": "1oAO5S1ikjxbbgPzBhZJf7Xf9bodbwwCE",

}

# Features in Subreddit
AUTHOR = 'author'
AUTHOR_FULLNAME = 'author_fullname'
CREATED_UTC = 'created_utc'
ID = 'id'
IDX = 'idx'
MAX_TIMESTAMP = 'max_timestamp'
MIN_TIMESTAMP = 'min_timestamp'
NETLOC = 'netloc'
POL_LEAN = 'pol_lean'
POST_ID = 'post_id'
RESOURCE = "resource"
SR_AUTHOR_TIME = "sr_author_time"
SUBREDDIT = 'subreddit'
UPDATED = 'updated'
TIME_DIFF = 'time_diff'
TIMESTAMP = 'timestamp'

# Types of resources
URL = 'url'  # Any types of URLs
V = 'v'  # YouTube Videos
MISINFORMATION = 'misinformation'  # URLs from misinformative domains as from FACTOID
ANYTHING = 'anything'  # Any types of posts

IDX_AUTHOR = 'idx_author'
IDX_INTERACTION = 'idx_interaction'
IDX_NODE = 'idx_node'
IDX_RESOURCE = 'idx_resource'
IDX_SUBREDDIT = 'idx_subreddit'
IDX_USER = 'idx_user'
IDX_SNAPSHOT = 'idx_snapshot'

SRC = "src"
DST = "dst"
SRC_RELABEL = "src_relabel"
DST_RELABEL = "dst_relabel"
NODE = "node"
RELATION = "relation"
RANKING = "ranking"
SCORE = "score"

POS_ITEMS = "pos_items"
NEG_ITEMS = "neg_items"

PRED = "pred"
LABEL = "label"
Y_PRED = "y_pred"
Y_TRUE = "y_true"
EVAL_INDEX = "eval_index"

# Sampling method
RANDOM = 'random'  # In use. Fast but not recommended
EXCLUDE_POSITIVE = 'exclude_positive'  # In use. Slow but recommended
PER_INTERACTION = 'per_interaction'

# Evaluation sampling mode
ALL = 'all'
SAMPLE = 'sample'

# train/val/test split
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

# Evaluation metrics
PRECISION = 'precision'
RECALL = 'recall'
F1 = 'f1'
ACCURACY = 'accuracy'
NDCG = 'ndcg'
TP = 'tp'
FP = 'fp'
TN = 'tn'
FN = 'fn'

# Method names or types of embeddings
APPNP = 'appnp'
CTGCN = 'ctgcn'
DYSAT = 'dysat'
GAT = 'gat'
GATEDGRAPHCONV = 'gatedgraphconv'
GCN = 'gcn'
LIGHTGCN = 'lightgcn'
LINE = 'line'
MF = 'mf'
NODE2VEC = 'node2vec'
SOCIALLSTM = 'sociallstm'
SRGNN = 'srgnn'
TGN = 'tgn'
TISASREC = 'tisasrec'

# Visualization Models
PCA = 'pca'
TSNE = 'tsne'
UMAP = 'umap'
ISOMAP = 'isomap'
MDS = 'mds'
LLE = 'lle'

UNK = 'unk'

# Types of word embeddings
# Specified in args.session_graph_operator
GLOVE = 'glove'
BERT = 'bert'

# How to split the interactions into different sessions, i.e. How to define a session
# Specified in args.session_split_method
SEQUENTIAL = 'sequential'
BIMODAL = 'bimodal'
OURS = 'ours'
SESSION = 'session'  # Split the session into multiple sessions
ALL = 'all'
dataset_names_36_months = []
dataset_names_54_months = []
dataset_names_60_months = []
dataset_names_120_months = []

for year in range(2013, 2023):
    dataset_names_120_months += [f'{year}-{str(i).zfill(2)}' for i in
                                 range(1, 13)]

for year in range(2018, 2022):
    dataset_names_54_months += [f'{year}-{str(i).zfill(2)}' for i in
                                range(1, 13)]
    dataset_names_60_months += [f'{year}-{str(i).zfill(2)}' for i in
                                range(1, 13)]

dataset_names_54_months += [f'2022-{str(i).zfill(2)}' for i in range(1, 7)]
dataset_names_60_months += [f'2022-{str(i).zfill(2)}' for i in range(1, 13)]

for year in range(2020, 2023):
    dataset_names_36_months += [f'{year}-{str(i).zfill(2)}' for i in
                                range(1, 13)]

dataset_names_3_months = [f'2020-{str(i).zfill(2)}' for i in range(1, 4, 1)]
dataset_names_12_months = [f'2020-{str(i).zfill(2)}' for i in range(1, 13, 1)]

dataset_names_1_months = [f'2020-{str(i).zfill(2)}' for i in range(1, 2, 1)]

# `54_months` is the full dataset used in the KDD2023 paper
# `60_months` is the full dataset used in the ICWSM2024 project

MONTH_ALL = "60_months"  # "60_months", "54_months"

dataset_name2months = {
    "3_months": dataset_names_3_months,
    "12_months": dataset_names_12_months,
    "36_months": dataset_names_36_months,
    "54_months": dataset_names_54_months,
    "60_months": dataset_names_60_months,
    "120_months": dataset_names_120_months,
}
dataset_name2months['60_months_sport_city'] = dataset_name2months['60_months_city_university'] = dataset_name2months['60_months']


"""
r/politics: This subreddit is a general forum for discussing political news and analysis. It covers a wide range of topics, including national and international politics, elections, and policy issues.

r/Liberal: This subreddit is geared towards liberal and progressive viewpoints, and is a good place to discuss progressive politics and social issues.

r/Conservative: This subreddit is geared towards conservative viewpoints, and is a good place to discuss conservative politics and social issues.

r/Anarchism: This subreddit is a forum for discussing anarchism and related topics, including political theory, philosophy, and current events.

r/LateStageCapitalism: This subreddit is a forum for discussing the problems with capitalism and advocating for alternative economic systems.

r/PoliticalDiscussion: This subreddit is a general forum for discussing political news, analysis, and opinion from a variety of viewpoints.

r/PoliticalHumor: This subreddit is a forum for discussing and sharing political jokes, memes, and other humorous content related to politics.

r/worldpolitics: This subreddit is a forum for discussing international politics and current events.

r/PoliticalCompassMemes: This subreddit is a forum for discussing and sharing memes related to the Political Compass, a tool for mapping political views.

r/PoliticalVideo: This subreddit is a forum for sharing and discussing political videos and documentaries.

r/PoliticalDiscourse: This subreddit is a forum for discussing and debating political issues and ideas in a civil manner.

r/PoliticalFactChecking: This subreddit is a forum for fact-checking and discussing political claims and statements.

r/PoliticalRevisionism: This subreddit is a forum for discussing and reexamining historical events and political ideologies.

r/PoliticalIdeology: This subreddit is a forum for discussing and debating different political ideologies and theories.

r/PoliticalRevolution: This subreddit is a forum for discussing and advocating for political change and revolution.

r/PoliticalMemes: This subreddit is a forum for sharing and discussing memes related to politics.

r/PoliticalModeration: This subreddit is a forum for discussing and advocating for moderation in political views and discourse.

r/PoliticalCorrectness: This subreddit is a forum for discussing and debating the concept of political correctness and its impact on society.

r/PoliticalCorrectnessGoneMad: This subreddit is a forum for discussing and sharing examples of what users see as excessive political correctness.

r/PoliticalTheory: This subreddit is a forum for discussing and debating political theory and ideas.

r/PoliticalQuestions: This subreddit is a forum for asking and answering questions about politics and current events.

r/PoliticalScience: This subreddit is a forum for discussing and debating the study of politics and government.

r/PoliticalHumorModerated: This subreddit is a moderated forum for sharing and discussing political jokes and memes.

r/PoliticalCorrectnessSucks: This subreddit is a forum for discussing and sharing examples of what users see as excessive political correctness.

r/PoliticalCompass: This subreddit is a forum for discussing and sharing content related to the Political Compass, a tool for mapping political views.

r/PoliticalDiscussionModerated: This subreddit is a moderated forum for discussing and debating political news, analysis, and opinion from a variety of viewpoints.

r/PoliticalDiscussionUK: This subreddit is a forum for discussing and debating political issues and events in the United Kingdom.

r/PoliticalDiscussionCanada: This subreddit is a forum for discussing and debating political issues and events in Canada.

r/PoliticalDiscussionAustralia: This subreddit is a forum for discussing and debating political issues and events in Australia.

r/PoliticalDiscussionEurope: This subreddit is a forum for discussing and debating political issues and events in Europe.

r/PoliticalDiscussionUSA: This subreddit is a forum for discussing and debating political issues and events in the United States.

r/PoliticalDiscussionAsia: This subreddit is a forum for discussing and debating political issues and events in Asia.

r/PoliticalDiscussionAfrica

"""

# Subreddits about politics

SUBREDDITS_ABOUT_POLITICS = ['politics', 'Liberal', 'Conservative', 'Anarchism',
                             'LateStageCapitalism', 'PoliticalDiscussion',
                             'PoliticalHumor', 'worldpolitics',
                             'PoliticalCompassMemes', 'PoliticalVideo',
                             'PoliticalDiscourse', 'PoliticalFactChecking',
                             'PoliticalRevisionism', 'PoliticalIdeology',
                             'PoliticalRevolution', 'PoliticalMemes',
                             'PoliticalModeration', 'PoliticalCorrectness',
                             'PoliticalCorrectnessGoneMad', 'PoliticalTheory',
                             'PoliticalQuestions', 'PoliticalScience',
                             'PoliticalHumorModerated',
                             'PoliticalCompass',
                             'PoliticalDiscussionModerated',
                             'worldnews', 'news', 'worldpolitics',
                             'worldevents', 'business', 'economics',
                             'environment', 'energy', 'law', 'education',
                             'history', 'PoliticsPDFs', 'WikiLeaks', 'SOPA',
                             'NewsPorn', 'worldnews2', 'AnarchistNews',
                             'republicofpolitics', 'LGBTnews', 'politics2',
                             'economic2', 'environment2', 'uspolitics',
                             'AmericanPolitics', 'AmericanGovernment',
                             'ukpolitics', 'canada', 'euro', 'Palestine',
                             'eupolitics', 'MiddleEastNews', 'Israel', 'india',
                             'pakistan', 'china', 'taiwan', 'iran', 'russia',
                             'Libertarian', 'Anarchism', 'socialism',
                             'progressive', 'Conservative',
                             'americanpirateparty', 'democrats', 'Liberal',
                             'new_right', 'Republican', 'egalitarian',
                             'demsocialist', 'LibertarianLeft', 'Liberty',
                             'Anarcho_Capitalism', 'alltheleft', 'neoprogs',
                             'democracy', 'peoplesparty', 'Capitalism',
                             'Anarchist', 'feminisms', 'republicans',
                             'Egalitarianism', 'anarchafeminism', 'Communist',
                             'socialdemocracy', 'conservatives', 'Freethought',
                             'StateOfTheUnion', 'equality', 'propagandaposters',
                             'SocialScience', 'racism', 'corruption',
                             'propaganda', 'lgbt', 'feminism', 'censorship',
                             'obama', 'war', 'antiwar', 'climateskeptics',
                             'conspiracyhub', 'infograffiti', 'CalPolitics',
                             'politics_new'
                             ]

# Subreddits preprocess for dataset crawling

FIELDS_SUBMISSION = [
    'author_fullname',
    'author_is_blocked',
    'banned_at_utc',
    'banned_by',
    'can_gild',
    'can_mod_post',
    'category',
    'content_categories',
    'created_utc',
    'domain',
    'downs',
    'fullname',
    'gilded',
    'gildings',
    'hidden',
    'id',
    'is_created_from_ads_ui',
    'is_crosspostable',
    'is_meta',
    'is_original_content',
    'is_reddit_media_domain',
    'is_robot_indexable',
    'is_self',
    'is_video',
    'locked'
    'name',
    'num_comments',
    'num_crossposts',
    'num_duplicates',
    'num_reports',
    'over_18',
    'parent_whitelist_status',
    'permalink',
    'pinned',
    'quarantine',
    'removal_reason',
    'removed_by',
    'removed_by_category',
    'report_reasons',
    'saved',
    'score',
    'selftext',
    'send_replies',
    'shortlink',
    'spoiler',
    'stickied',
    'subreddit_id',
    'subreddit_name_prefixed',
    'thumbnail',
    'title',
    'ups',
    'upvote_ratio',
    'url',
    'view_count',
    'wls',
]

PATH_GLOVE = "F:\\dataset\\NLP\\GloVe\\glove.6B.100d.txt"
PATH_DYSAT = "D:\\Workspace\\Graph\\DySAT"


if platform.system() == 'Windows':
    PATH_REDDIT_COMMENTS_15_YEARS = "E:\\data\\Reddit\\comments"
    PATH_REDDIT_WEEKLY_COMMENTS_15_YEARS = "E:\\data\\Reddit\\weekly_comments"

else:
    PATH_REDDIT_COMMENTS_15_YEARS = "/nethome/yjin328/Workspace/data/FakeNews/Reddit/comments"
    PATH_REDDIT_WEEKLY_COMMENTS_15_YEARS = "/nethome/yjin328/Workspace/data/FakeNews/Reddit/weekly_comments"

# Get a logo on https://patorjk.com/software/taag/

CLAWS_LAB = """
  #####  #          #    #     #  #####  
 #     # #         # #   #  #  # #     # 
 #       #        #   #  #  #  # #       
 #       #       #     # #  #  #  #####  
 #       #       ####### #  #  #       # 
 #     # #       #     # #  #  # #     # 
  #####  ####### #     #  ## ##   #####                                                                         
"""

DYGETVIZ= """
  _____         _____ ______ _________      ___     
 |  __ \       / ____|  ____|__   __\ \    / (_)    
 | |  | |_   _| |  __| |__     | |   \ \  / / _ ____
 | |  | | | | | | |_ |  __|    | |    \ \/ / | |_  /
 | |__| | |_| | |__| | |____   | |     \  /  | |/ / 
 |_____/ \__, |\_____|______|  |_|      \/   |_/___|
          __/ |                                     
         |___/                                      
"""


REDDIT = 'reddit'

WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""

CONFERENCE_LIST = ['AAAI',
                   'AAMAS',
                   'ACCV',
                   'ACL',
                   'ACM DL',
                   'ACM Multimedia',
                   'AIRS',
                   'AISTATS',
                   'ALT',
                   'ALTA',
                   'AMCIS',
                   'BMVC',
                   'CHI',
                   'CHIIR',
                   'CIKM',
                   'COLD',
                   'COLING',
                   'COLT',
                   'CSCW',
                   'CVPR',
                   'CoNLL',
                   'COLING',
                   'EACL',
                   'ECCV',
                   'ECDL',
                   'ECIR',
                   'ECML',
                   'ECML-PKDD',
                   'EDBT',
                   'EMNLP',
                   'ESWC',
                   'EuroVis',
                   'Eurographics',
                   'GROUP',
                   'HCOMP',
                   'HLT',
                   'ICASSP',
                   'ICB',
                   'ICC',
                   'ICCBR',
                   'ICCC',
                   'ICCV',
                   'ICDAR',
                   'ICDE',
                   'ICDM',
                   'ICDT',
                   'ICER',
                   'ICHR',
                   'ICIP',
                   'ICLR',
                   'ICME',
                   'ICMI',
                   'ICML',
                   'ICMLA',
                   'ICMR',
                   'ICPR',
                   'ICPRAM',
                   'ICRA',
                   'ICSLP',
                   'ICTAI',
                   'ICTIR',
                   'ICWSM',
                   'IIR',
                   'IJCAI',
                   'IJCNN',
                   'IMWUT',
                   'INTER-SPEECH',
                   'INTERSPEECH',
                   'IROS',
                   'ISM',
                   'ISMAR',
                   'ISWC',
                   'JCDL',
                   'KDD',
                   'KR',
                   'LREC',
                   'MMSys',
                   'MobiSys',
                   'NAACL',
                   'NAACL-HLT',
                   'NIPS',
                   'NeurIPS',
                   'PAKDD',
                   'PCM',
                   'PKDD',
                   'RANLP',
                   'RecSys',
                   'SAC',
                   'SDM',
                   'SEMANTiCS',
                   'SIBGRAPI',
                   'SIGAI',
                   'SIGGRAPH',
                   'SIGIR',
                   'SIGMOD',
                   'SIGKDD',
                   'SocInfo',
                   'TREC',
                   'UAI',
                   'UMAP',
                   'Ubicomp',
                   'VISAPP',
                   'VLDB',
                   'WSDM',
                   'WWW',
                   'WebConf',
                   'WebDB',
                   'WebSci']

ARXIV_CATEGORIES = ['cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL',
                    'cs.CR',
                    'cs.CV', 'cs.CY', 'cs.DB', 'cs.DC',
                    'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL', 'cs.GL',
                    'cs.GR', 'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT',
                    'cs.LG', 'cs.LO', 'cs.MA', 'cs.ML', 'cs.MM', 'cs.MS',
                    'cs.NA', 'cs.NE', 'cs.NI', 'cs.OH', 'cs.OS', 'cs.PF',
                    'cs.PL', 'cs.RO', 'cs.SC', 'cs.SD', 'cs.SE',
                    'cs.SI', 'cs.SY', 'stat.ML']
