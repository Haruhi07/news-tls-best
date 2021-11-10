import os
import json
import shutil
import pathlib

data_dir = pathlib.Path('./dataset/Timeline17')
#os.mkdir('./processed_dataset')
timeline_out_dir = pathlib.Path('./dataset/Timeline17')
out_dir = data_dir
topics = os.listdir(data_dir)
f_tmp = open("cov_log.txt", "w")

for topic in topics:
    if topic == ".DS_Store":
        continue
    print("topic = {}".format(topic))
    #os.mkdir(os.path.join(out_dir, topic))
    article_json = open(os.path.join(out_dir, topic, "articles.jsonl"), "w")
    dates_dir = data_dir/topic/'InputDocs'
    dates = os.listdir(dates_dir)
    for date in dates:
        if date == ".DS_Store":
            continue
        article_set_dir = os.path.join(dates_dir, date)
        articles = os.listdir(article_set_dir)
        for article in articles:
            article_dir = os.path.join(article_set_dir, article)
            article_file = open(article_dir, "r")
            text = article_file.read()
            id = article[:-8]
            data = {"id" : id, "time" : date, "text" : text}
            json_data = json.dumps(data)
            article_json.write(json_data + "\n")
    article_json.close()

    timeline_json = open(timeline_out_dir/topic/"timelines.jsonl", "w")
    raw_timeline_dir = data_dir/topic/"timelines"
    timelines = os.listdir(raw_timeline_dir)
    timeline_list = []
    for timeline in timelines:
        print(timeline)
        if timeline == ".DS_Store":
            continue
        timeline_file = open(os.path.join(raw_timeline_dir, timeline), "r")
        timeline_text = timeline_file.readlines()
        text_list = []
        time_ = ""
        for line in timeline_text:
            print("line = {}".format(line), file=f_tmp)
            tmp = line.split('-')
            if len(tmp) == 3 and len(tmp[0]) == 4 and len(tmp[1]) == 2 and len(tmp[2]) == 3:
                time_ = line.replace("\n", "")
                print("time", file=f_tmp)
            elif len(tmp) == 33:
                aligned_timelines = (time_, text_list)
                timeline_list.append(aligned_timelines)
                #tmp_json = json.dumps(aligned_timelines)
                #timeline_json.write(tmp_json + "\n")
                text_list = []
                time = ""
                print("---", file=f_tmp)
            else:
                text_list.append(line)
                print("text", file=f_tmp)
        tmp_json = json.dumps(timeline_list)
        timeline_json.write(tmp_json + "\n")
        timeline_file.close()
    timeline_json.close()

    keywords_json = open(os.path.join(out_dir, topic, "keywords.json"), "w")
    keywords = [topic.split('_')[0]]
    json_data = json.dumps(keywords)
    keywords_json.write(json_data + "\n")
    keywords_json.close()

f_tmp.close()
