import argparse
import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

ids = ["A","B","C","D","E"]


design = [
[1,"future_happy_week","a future week that could feel happier and more energising","low","Use clear, direct, minimally embellished language."],
[2,"future_happy_week","a future week that could feel happier and more energising","high","Use moderately vivid and expressive language while remaining clear and easy to follow."],
[3,"future_recovery_week","a future week focused on recovery and balancing activity with rest","low","Use clear, direct, minimally embellished language."],
[4,"future_recovery_week","a future week focused on recovery and balancing activity with rest","high","Use moderately vivid and expressive language while remaining clear and easy to follow."],
[5,"past_importance","whether activities during the past week were important or meaningful","low","Use clear, direct, minimally embellished language."],
[6,"past_importance","whether activities during the past week were important or meaningful","high","Use moderately vivid and expressive language while remaining clear and easy to follow."],
[7,"past_motives","whether the reasons for activities during the past week were fulfilled","low","Use clear, direct, minimally embellished language."],
[8,"past_motives","whether the reasons for activities during the past week were fulfilled","high","Use moderately vivid and expressive language while remaining clear and easy to follow."]
]

system_prompt = """You write short, supportive wellbeing narratives for older adults.
Follow the requested topic and format. Do not provide medical diagnosis or treatment.
Return only the narrative, without a title, headings, labels, bullets, or commentary."""

def make_prompt(x):
 story,topic_id,topic,creativity,style=x
 return f"""Write a short narrative about {topic}.

Write in second person ("you").
Tone: supportive, factual, and suitable for older adults.
Use clear, simple language. Avoid slang and exaggerated imagery.
{style}
Target length: 150-180 words (maximum 180 words).

Use exactly three paragraphs:
1. A warm opening that introduces the topic without inventing personal facts.
2. A development of the topic in a general, non-person-specific way.
3. A brief closing that reinforces reflection or practical consideration.

Do not assume or invent the reader's name, activities, preferences, priorities,
goals, experiences, health conditions, relationships, location, or past actions.
Do not mention a persona, user model, dialogue type, argument, premise, claim,
argumentation scheme, knowledge graph, or grounding instruction."""

def hash_text(x):
 return hashlib.sha256(x.encode("utf-8")).hexdigest()

def count_words(x):
 return len(re.findall(r"\b[\w’'-]+\b",x))

def check_text(x):
 wc=count_words(x)
 ps=[a.strip() for a in re.split(r"\n\s*\n",x.strip()) if a.strip()]
 return {
 "word_count":wc,
 "paragraph_count":len(ps),
 "length_ok":150<=wc<=180,
 "three_paragraphs_ok":len(ps)==3,
 "second_person_present":bool(re.search(r"\b(you|your|you’re|you've|you’ll)\b",x,re.I))
 }

def get_args():
 p=argparse.ArgumentParser()
 p.add_argument("output-dir",default="results/condition_c_standard_llm")
 p.add_argument("model",default="gpt-5-mini")
 p.add_argument("request-delay",type=float,default=0.2)
 p.add_argument("max-api-attempts",type=int,default=5)
 p.add_argument("overwrite",action="store_true")
 return p.parse_args()

def call_api(client,model,item,temp):
 return client.responses.create(
 model=model,
 instructions=system_prompt,
 input=make_prompt(item),
 temperature=temp
 )

def call_with_retry(client,model,item,temp,max_attempts):
 last=None
 for i in range(max_attempts):
  try:
   return call_api(client,model,item,temp),i+1
  except Exception as e:
   last=e
   if i<max_attempts-1:
    time.sleep(min(2**i,20))
 raise RuntimeError("API request failed: "+str(last))

def main():
 args=get_args()

 if not os.getenv("OPENAI_API_KEY"):
  print("OPENAI_API_KEY is not set")
  return

 out=Path(args.output_dir)
 out.mkdir(parents=True,exist_ok=True)

 csvfile=out/"condition_c_narratives.csv"
 jsonfile=out/"condition_c_raw_responses.jsonl"
 metafile=out/"run_metadata.json"
 manifestfile=out/"condition_c_prompt_manifest.json"

 if not args.overwrite:
  if csvfile.exists() or jsonfile.exists() or metafile.exists():
   print("Output already exists. Use overwrite or another folder.")
   return

 manifest=[]
 for x in design:
  story,topic_id,topic,creativity,style=x
  prompt=make_prompt(x)
  manifest.append({
  "story":story,
  "topic_id":topic_id,
  "topic":topic,
  "creativity":creativity,
  "style_instruction":style,
  "temperature":temps[creativity],
  "user_prompt":prompt,
  "prompt_sha256":hash_text(system_prompt+"\n"+prompt)
  })

 manifestfile.write_text(json.dumps({"system_prompt":system_prompt,"design":manifest},indent=2,ensure_ascii=False),encoding="utf-8")

 client=OpenAI()
 rows=[]
 returned_models=set()

 f=open(jsonfile,"w",encoding="utf-8")

 for matched_id in ids:
  for x in design:
   story,topic_id,topic,creativity,style=x
   temp=temps[creativity]

   response,attempts=call_with_retry(client,args.model,x,temp,args.max_api_attempts)
   text=response.output_text.strip()
   checks=check_text(text)
   returned_model=getattr(response,"model",None)

   if returned_model:
    returned_models.add(returned_model)

   row={
   "condition":"C",
   "matched_id":matched_id,
   "story":story,
   "topic_id":topic_id,
   "creativity":creativity,
   "text":text,
   "word_count":checks["word_count"],
   "paragraph_count":checks["paragraph_count"],
   "length_ok":checks["length_ok"],
   "three_paragraphs_ok":checks["three_paragraphs_ok"],
   "second_person_present":checks["second_person_present"],
   "requested_model":args.model,
   "returned_model":returned_model,
   "temperature":temp,
   "response_id":getattr(response,"id",None),
   "api_attempts":attempts,
   "prompt_sha256":hash_text(system_prompt+"\n"+make_prompt(x))
   }

   rows.append(row)
   f.write(json.dumps(row,ensure_ascii=False)+"\n")
   f.flush()
   time.sleep(args.request_delay)

 f.close()

 with open(csvfile,"w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
  w.writeheader()
  w.writerows(rows)

 metadata={
 "script_version":version,
 "created_utc":datetime.now(timezone.utc).isoformat(),
 "n_narratives":len(rows),
 "requested_model":args.model,
 "returned_models":sorted(returned_models),
 "temperature_schedule":{
 "low_creativity":0.2,
 "high_creativity":0.9,
 "low_creativity_stories":[1,3,5,7],
 "high_creativity_stories":[2,4,6,8]
 },
 "selection_policy":"One successful API response per matched item; no selective regeneration for content/QC.",
 "excluded_inputs":[
 "persona/user-model information",
 "activity data",
 "grounding instructions",
 "dialogue-type labels or guidance",
 "argumentation-scheme guidance"
 ],
 "qc_summary":{
 "length_ok":sum(1 for r in rows if r["length_ok"]),
 "three_paragraphs_ok":sum(1 for r in rows if r["three_paragraphs_ok"]),
 "second_person_present":sum(1 for r in rows if r["second_person_present"])
 }
 }

 metafile.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
 print(json.dumps(metadata,indent=2))
 print("Outputs saved in:",out.resolve())

if __name__=="__main__":
 main()
