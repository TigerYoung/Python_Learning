from selenium import webdriver
from bs4 import BeautifulSoup
import re
import time
from selenium.webdriver.common.keys import Keys
driver = webdriver.PhantomJS()
driver.get('https://www.quora.com/Is-Python-a-dying-language')
time.sleep(1)
for i in range(100):
    driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    time.sleep(1)
soup = BeautifulSoup(driver.page_source, 'lxml')
contents = soup.find_all(id=re.compile("__w2_([0-9]|[a-z]|[A-Z]){7}_answer_content"))
text = []
print len(contents)

for i in range(len(contents)):
    content = ""
    for child in contents[i].descendants:
        try:
            if child.child is not None:
                continue
        except:
            #print child
            content += child
    #print content
    content = content.encode('ascii', 'ignore').decode('ascii')
    Quora_file = open("./output/quorafile" + str(i) + '.txt', "w")
    Quora_file.write(str(content))