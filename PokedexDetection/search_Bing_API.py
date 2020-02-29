#import all necessary packages
from requests import exceptions
import argparse
import cv2
import requests, json
import os

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required = True, help = "query to search Bing search API")
ap.add_argument("-o", "--output", required= True, help = "path for the output images")
args = vars(ap.parse_args())

#set Microsoft Azure service details, then set maximum no of results for given search and group size for results
API_KEY = "1193f3494a754fa6b9dee2851387ceeb"
MAX_RESULTS = 250
GROUP_SIZE = 50

#set the endpoint API results
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

#build filter for exceptions
EXCEPTIONS = set([IOError, FileNotFoundError,exceptions.RequestException, exceptions.HTTPError,exceptions.ConnectionError, exceptions.Timeout])

#store the search term and set headers and search parameters
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key": API_KEY}
params = {"q":term,"offset":0,"count": GROUP_SIZE}

#make the search
print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL,headers=headers,params=params)
search.raise_for_status()

#grab the results from the search and estimate the results
results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total estmated results {}".format(estNumResults, term))

#initialize the total number of images downloaded so far
total = 0

#loop over estimated number of results in group GROUP_SIZE
for offset in range(0, estNumResults, GROUP_SIZE):
    #update the search parameters with current offset and make the request fetch the results
    print("[INFO ] making request group of {}-{} of {}".format(offset, offset+GROUP_SIZE,estNumResults))
    params["offset"] = offset
    search = requests.get(URL, headers = headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO ] making request group of {}-{} of {}".format(offset, offset+GROUP_SIZE,estNumResults))

#loop overthe results
for v in results["value"]:
    #try to download the images
    try:
        print("[INFO] fetching: {}".format(v["contentUrl"]))
        r = requests.get(v["contentUrl"], timeout = 30)

        #build the content of the output
        ext = v["contentUrl"][v["contentUrl"].rfind("."):]
        p = os.path.sep.join([args["output"], "{}{}".format(str(total).zfill(8), ext)])

        #write the image to disk
        f = open(p, "wb")
        f.write(r.content)
        f.close()

    #check for any exceptions that occur during download
    except Exception as e:
        if type(e) in EXCEPTIONS:
            print("[INFO] skipping: {}".format(v["contentUrl"]))
            continue

        #load the image fron disk
    image = cv2.imread(p)
        #if image is None then we didn't load the image properly
    if image is None:
        print("[INFO] deleting: {}".format(p))
        os.remove(p)
        continue

        #increment the total counter
    total +=1
