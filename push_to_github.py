"""
BSNL Fault Prediction — GitHub Auto-Push Script
Run: python push_to_github.py
"""
import subprocess, sys, os, json, urllib.request, urllib.error

GITHUB_USERNAME = "enggbikash15"
REPO_NAME       = "bsnl-fault-prediction"
REPO_DESC       = "BSNL Fibre Fault Prediction System — ML pipeline for telecom OFC fault prediction"
PRIVATE         = False

# Paste your GitHub Personal Access Token here
# Get from: github.com → Settings → Developer Settings → Personal Access Tokens → Fine-grained
GITHUB_PAT = "PASTE_YOUR_PAT_HERE"

def run(cmd, cwd=None, check=True):
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.stdout.strip(): print(f"    {r.stdout.strip()}")
    if r.returncode != 0 and check:
        print(f"  [ERROR] {r.stderr.strip()}"); sys.exit(1)
    return r

def create_repo(pat, user, name, desc, private):
    try:
        payload = json.dumps({"name":name,"description":desc,"private":private,"auto_init":False}).encode()
        req = urllib.request.Request("https://api.github.com/user/repos", data=payload,
            headers={"Authorization":f"token {pat}","Accept":"application/vnd.github.v3+json",
                     "Content-Type":"application/json","User-Agent":"Python-Script"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            return data.get("html_url"), None
    except urllib.error.HTTPError as e:
        body = json.loads(e.read().decode()); msg = body.get("message","")
        if "already exists" in msg.lower() or e.code == 422:
            return f"https://github.com/{user}/{name}", "exists"
        return None, msg
    except Exception as e:
        return None, str(e)

def main():
    print("\n" + "="*60)
    print("  BSNL FAULT PREDICTION — GITHUB AUTO-PUSH")
    print("="*60)

    if GITHUB_PAT == "PASTE_YOUR_PAT_HERE":
        print("\n[ERROR] Set your GitHub PAT in this script first.")
        print("  1. Go to: github.com → Settings → Developer Settings → Personal Access Tokens")
        print("  2. Generate token with 'repo' scope")
        print("  3. Paste in GITHUB_PAT variable above\n")
        sys.exit(1)

    print("\n[1/4] Creating GitHub repository...")
    url, err = create_repo(GITHUB_PAT, GITHUB_USERNAME, REPO_NAME, REPO_DESC, PRIVATE)
    if err == "exists":  print(f"  Repo already exists → {url}")
    elif err:            print(f"  [ERROR] {err}"); sys.exit(1)
    else:                print(f"  Created → {url}")

    d = os.path.dirname(os.path.abspath(__file__))

    print("\n[2/4] Initialising Git...")
    run("git init", cwd=d)
    run('git config user.email "enggbikash15@github.com"', cwd=d)
    run('git config user.name "Bikash"', cwd=d)

    print("\n[3/4] Staging & committing...")
    run("git add .", cwd=d)
    run('git commit -m "Initial commit: BSNL Fibre Fault Prediction ML System"', cwd=d)
    run("git branch -M main", cwd=d)

    print("\n[4/4] Pushing to GitHub...")
    remote = f"https://{GITHUB_USERNAME}:{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
    run("git remote remove origin", cwd=d, check=False)
    run(f"git remote add origin {remote}", cwd=d)
    run("git push -u origin main", cwd=d)

    print("\n" + "="*60)
    print(f"  SUCCESS!")
    print(f"  URL: https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
