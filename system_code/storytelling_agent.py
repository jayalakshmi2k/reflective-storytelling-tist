import os
from openai import OpenAI
from app.handlers.activity_data_handler import ActivityDataHandler

client=OpenAI()
MODEL=os.getenv("STORY_MODEL","gpt-5-mini")

schemes={
"ArgumentFromValue":{
"cqs":[
"Are the stated values genuinely held?",
"Does the activity really promote the goal here?",
"Are there conflicts with other important values?",
"Is the activity necessary or sufficient? Are there better alternatives?"
]},
"PositionToKnow":{
"cqs":[
"Is the source in a position to know?",
"Is the source reliable and unbiased?",
"Is there contrary information?",
"Is this consistent with known facts?"
]},
"SufficientCondition":{
"cqs":[
"Is the activity really sufficient under the current circumstances?",
"Are there negative side effects?",
"Are there safer or easier alternatives?",
"Is there evidence for the conditional link?"
]}
}

def p(role,text):
 return {"role":role,"text":text}

def format_rows(rows):
 lines=[]
 for r in rows:
  bits=[]
  if r.get("importance") not in ["",None]:
   bits.append("importance "+str(r["importance"]))
  if r.get("fun_level") not in ["",None]:
   bits.append("fun "+str(r["fun_level"]))
  if r.get("frequency"):
   bits.append("frequency "+str(r["frequency"]))
  name=str(r.get("activity_name","")).strip()
  if not name:
   name="activity"
  if bits:
   lines.append("- "+name+" - "+", ".join(bits))
  else:
   lines.append("- "+name)
 return "\n".join(lines)

def activity_context(user_id,dialogue_type):
 h=ActivityDataHandler(user_id)

 if dialogue_type.lower()=="persuasion":
  pools=[h.fetch_important_data(),h.fetch_fun_data()]
 elif dialogue_type.lower()=="deliberation":
  pools=[h.fetch_recovery_data(),h.fetch_social_data(),h.fetch_priority_mix(top_n=2)]
 else:
  pools=[h.fetch_motivation_data(),h.fetch_important_data(),h.fetch_priority_mix(top_n=2)]

 out={}
 keys=["activity_name","frequency","fun_level","importance","done_with","motivation"]

 for row in pools:
  if not row:
   continue
  for k in keys:
   if k not in row or not row[k]:
    continue
   value=str(row[k])
   if "," in value:
    vals=[x.strip() for x in value.split(",") if x.strip()]
   else:
    vals=[x.strip() for x in value.split("|") if x.strip()]
   if k not in out:
    out[k]=[]
   for x in vals:
    if x not in out[k]:
     out[k].append(x)

 for k in out:
  out[k]=out[k][:2]

 return out

def style(creativity):
 if creativity.lower()=="low":
  return "STYLE: Plain language. Start with a positive, validating line. Factual, supportive, concise."
 return "STYLE: Engaging but concise. Start with a positive, validating line. Factual and supportive. Keep imagery minimal."

class StorytellingAgent:
 def __init__(self,user_data):
  self.user_data=user_data

 def reason(self,dialogue_type,data,story_prompt=""):
  activity=(data.get("activity_name") or "an activity").strip()
  freq=data.get("frequency")
  imp=data.get("importance")

  if dialogue_type.lower()=="persuasion":
   s="ArgumentFromValue"
   premises=[
   p("Goal","You want the coming week to be happier and energizing."),
   p("Values","Well-being and enjoyment matter to you."),
   p("Nec/Suff-For-Goal","Bringing about "+activity.lower()+" in small, safe doses supports your goal.")
   ]
   conclusion="Therefore, you should plan small, enjoyable, safe activities next week."

  elif dialogue_type.lower()=="deliberation":
   s="SufficientCondition"
   premises=[
   p("If-A-then-Goal","If you do "+activity.lower()+" on alternate days, you will recover better."),
   p("If-A-then-Value","Spacing effort protects energy and reduces stress."),
   p("Current-Circumstances","Recent pattern: "+activity+(" "+str(freq) if freq else "")+".")
   ]
   conclusion="Action: Adopt a lighter routine-"+activity.lower()+" on alternate days plus one quiet recovery block."

  elif dialogue_type.lower()=="inquiry":
   if "motives" in story_prompt.lower():
    s="ArgumentFromValue"
    premises=[
    p("Goal","Your motives were to act in line with your values."),
    p("Values","The activities you care about reflect those values."),
    p("Nec/Suff-For-Goal","Doing "+activity.lower()+(" "+str(freq) if freq else "")+" advances your valued goals.")
    ]
    conclusion="Therefore, your week aligned with your stated motives."
   else:
    s="PositionToKnow"
    obs="The system asserts you engaged in "+activity
    if imp:
     obs+=" rated "+str(imp)+"/5 in importance"
    if freq:
     obs+=" with frequency "+str(freq)
    premises=[
    p("Assertion",obs+"."),
    p("PositionToKnow","The system knows this from your activity records.")
    ]
    conclusion="Therefore, it is plausible that you did something important last week."

  else:
   s="SufficientCondition"
   premises=[p("Current-Circumstances","Limited data available.")]
   conclusion="Action: Keep a light, safe routine next week."

  return {
  "dialogue_type":dialogue_type,
  "scheme":s,
  "premises":premises,
  "conclusion":conclusion,
  "cqs_expected":schemes[s]["cqs"]
  }

 def render_plan(self,plan):
  premises=" ".join([x["text"] for x in plan["premises"]])
  cqs=" ".join(plan["cqs_expected"])
  return (
  "Dialogue Type: "+plan["dialogue_type"]+"\n"+
  "Argumentation Scheme: "+plan["scheme"]+"\n"+
  "Premises to consider:\n"+premises+"\n"+
  "Conclusion to guide the story: "+plan["conclusion"]+"\n"+
  "Reflect also on:\n"+cqs
  )

 def make_prompt(self,user,story_prompt,dialogue_type,context_type,plan_text="",use_scheme=True):
  activities=user.get("activities") or []
  acts=", ".join(activities) if isinstance(activities,list) else str(activities)

  h=ActivityDataHandler(user.get("user_id",0))
  mix=h.fetch_priority_mix(top_n=2) or {}
  rows=mix.get("rows",[])

  if rows:
   ratings="Rated/Logged activities:\n"+format_rows(rows)
  else:
   r=activity_context(user.get("user_id",0),dialogue_type)
   lines=[]
   if r.get("activity_name"):
    lines.append("Activities: "+", ".join(r["activity_name"]))
   if r.get("importance"):
    lines.append("Importance (1-5): "+", ".join(r["importance"]))
   if r.get("fun_level"):
    lines.append("Fun (1-5): "+", ".join(r["fun_level"]))
   if r.get("frequency"):
    lines.append("Frequencies: "+", ".join(r["frequency"]))
   ratings="\n".join(lines)

  bias=""
  if dialogue_type.lower() in ["inquiry","deliberation"]:
   bias=ratings+"\nSelection guidance:\n- Higher number = more important/fun; prioritize 4-5.\n- Include one lower-rated item only if useful.\n"

  if dialogue_type.lower()=="inquiry":
   para2="Paragraph 2 (4-6 sentences): Reflect on the past week. Summarize meaningful moments, values, or lessons from the activities. Mention what made them important or fulfilling."
   para3="Paragraph 3 (2-3 sentences): Do NOT propose a plan. Briefly reflect on what this story means for you and close with a warm, reassuring line."
  else:
   para2="Paragraph 2 (4-6 sentences): Transition to next week. Tie 2-3 suggestions to the rated activities. Use 'because' to connect suggestions to values."
   para3="Paragraph 3 (2-3 sentences): Offer a simple next-week plan linked to the narrative. End with an encouraging line."

  txt=(
  "Name: "+str(user.get("name","User"))+".\n"+
  "Activities: "+acts+".\n"+
  "Story Prompt: "+story_prompt+".\n"
  )

  if use_scheme:
   txt+="Argument Plan: "+plan_text+".\n"

  txt+=(
  "Dialogue Type: "+dialogue_type+".\n"+
  "Context Type: "+context_type+".\n"+
  bias+
  "Write a short narrative in second person ('you').\n"+
  "Tone: factual, supportive, optimistic, non-judgmental, and suitable for an older adult.\n"+
  "Use clear, simple language and avoid slang.\n"+
  "Style hygiene: prefer sentences <= 12 words; avoid metaphors and flowery language.\n"+
  "Length target: 150-180 words (hard cap 180 words).\n\n"+
  "Story shape (strict):\n"+
  "Paragraph 1 (2-3 sentences): Warm, validating opening that highlights effort or strengths. Start positive. Use a tiny scene from daily life to ground it.\n"+
  para2+"\n"+
  para3+"\n\n"+
  "Output requirements:\n"+
  "- Output exactly 3 paragraphs (blank line between paragraphs).\n"+
  "- Do not use headings, lists, labels, or schedules.\n"
  )
  return txt

 def generate(self,prompt,creativity):
  response=client.responses.create(
  model=MODEL,
  input=style(creativity)+"\n\n"+prompt
  )
  return response.output_text.strip()

 def generate_with_schemes(self,story_prompt,user_preferences,creativity,dialogue_type,context_type,data):
  acts=(data.get("activity_name") or "").strip()
  user=dict(user_preferences)
  user["activities"]=[x.strip() for x in acts.split(",") if x.strip()]

  plan=self.reason(dialogue_type,data,story_prompt)
  plan_text=self.render_plan(plan)
  prompt=self.make_prompt(user,story_prompt,dialogue_type,context_type,plan_text,True)
  story=self.generate(prompt,creativity)

  return {"story_text":story,"plan":plan,"prompt":prompt}

 def generate_without_schemes(self,story_prompt,user_preferences,creativity,dialogue_type,context_type,data):
  acts=(data.get("activity_name") or "").strip()
  user=dict(user_preferences)
  user["activities"]=[x.strip() for x in acts.split(",") if x.strip()]

  prompt=self.make_prompt(user,story_prompt,dialogue_type,context_type,"",False)
  story=self.generate(prompt,creativity)

  return {"story_text":story,"prompt":prompt}
