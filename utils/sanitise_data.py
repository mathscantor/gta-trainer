import pandas as pd

df = pd.read_csv("../datasets/gta-v1.csv")
domains = df["domain"].values
labels = df["class"].values

sanitised_domains = []
for domain in domains:
    domain = domain.replace("b'", "").strip()
    domain = domain.replace("'", "").strip()
    sanitised_domains.append(domain)

print(sanitised_domains)
print(labels)

f = open("../datasets/gta-v2.csv", "w")
f.write("domain,label\n")
if len(sanitised_domains) != len(labels):
    print("FAIL")
    exit()
for i in range(len(sanitised_domains)):
    f.write("{},{}\n".format(sanitised_domains[i], labels[i]))

