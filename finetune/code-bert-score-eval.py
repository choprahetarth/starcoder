import code_bert_score

def compute_similarity(code1, code2):
    # The score function returns a 4-tuple of (precision, recall, F1, F3)
    # We only need the F1 score for similarity, so we extract that
    _, _, f1_score, _ = code_bert_score.score(cands=[code1], refs=[code2], lang='python')
    return f1_score.item()  # Convert tensor to a standard Python number

# example usage
# code1 = """
# ---
# - hosts: your_host
#   become: yes
#   tasks:
#     - name: Update apt cache
#       apt:
#         update_cache: yes

#     - name: Install Nginx
#       apt:
#         name: nginx
#         state: present
# """

# code2 = """
# ---
# - hosts: your_host
#   become: yes
#   tasks:
#     - name: Update apt cache
#       shell: apt-get update

#     - name: Install Nginx
#       shell: apt-get install -y nginx
# """

# print(compute_similarity(code1, code2))