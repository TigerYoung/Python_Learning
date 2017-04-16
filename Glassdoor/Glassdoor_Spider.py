#coding:utf-8
import requests
import re
from bs4 import BeautifulSoup
proxies = {
  "http": "XXXXXXXX:8080",
  "https": "XXXXXXXX:8080",
}

def collectsite(site):
    req = requests.get(site, headers={'User-Agent': 'Mozilla/5.0'}, proxies=proxies)
    html_doc = req.text
    soup = BeautifulSoup(html_doc, "lxml")
    contents = soup.find_all(class_='logoWrap')
    result_findall = re.findall(r"\"/partner/jobListing.*?\"", str(contents))
    sites = []
    for item in result_findall:
        site1 = "https://www.glassdoor.com.au" + item[1:-1]
        site2 = site1.replace("amp;", "")
        sites.append(site2)
    return sites

def parsesite(site, num):
    req = requests.get(site, headers={'User-Agent': 'Mozilla/5.0'}, proxies=proxies)
    html_doc = req.text
    soup = BeautifulSoup(html_doc, "lxml")
    contents = soup.find_all(class_='jobDescriptionContent desc')
    result = contents[0].contents
    glassdoorfile = open('./glassdoorfile/glassdoorfile' + str(num) + '.txt', 'w')
    glassdoorfile.write(str(result))
    rate = soup.find_all(class_='ratingNum')
    rate_result = rate[0].contents
    return rate_result[0]

def main():
    allsites = []
    for i in range(1,34):
        site = 'https://www.glassdoor.com.au/Job/data-jobs-SRCH_KE0,4_IP' + str(i) + '.htm'
        allsites += collectsite(site)
    print len(allsites)
    rates = []
    for j in range(len(allsites)):
        try:
            rate = parsesite(allsites[j], j)
            rates.append(rate)
            #print parsesite(allsites[j], j)
            print "glassdoorfile" + str(j) + " generated!"
        except:
            continue
    print rates
    ratefile = open('./glassdoorrate.txt', 'w')
    ratefile.write(str(rates))

if __name__ == "__main__":
    main()

