#!/bin/bash
# ============================================================================
# DOMAIN 12: GIT & VERSION CONTROL
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "12 — GIT"
G="$PROV_WORKDIR/git_test"; mkdir -p "$G"

run "git init" git init "$G/repo"
run "git config" git -C "$G/repo" config user.email "prov@test.local" && git -C "$G/repo" config user.name "Prov Test"
echo "# Test" > "$G/repo/README.md"; echo "print('hello')" > "$G/repo/main.py"; echo "*.pyc" > "$G/repo/.gitignore"
run "git add -A" git -C "$G/repo" add -A
run "git status" git -C "$G/repo" status
run "git status -s" git -C "$G/repo" status -s
run "git commit" git -C "$G/repo" commit -m "Initial commit"
run "git log" git -C "$G/repo" log --oneline
run "git log --graph" git -C "$G/repo" log --graph --oneline --all
run "git log -p" git -C "$G/repo" log -p | head -20
run "git log --stat" git -C "$G/repo" log --stat
run "git log --format" git -C "$G/repo" log --format="%H %an %s"
run "git shortlog" git -C "$G/repo" shortlog -s
run "git branch" git -C "$G/repo" branch feature-1
run "git branch -a" git -C "$G/repo" branch -a
run "git checkout" git -C "$G/repo" checkout feature-1
echo "new feature" > "$G/repo/feature.txt"
run "git add" git -C "$G/repo" add feature.txt
run "git commit feature" git -C "$G/repo" commit -m "Add feature"
run "git checkout main" git -C "$G/repo" checkout main 2>/dev/null || git -C "$G/repo" checkout master
run "git merge" git -C "$G/repo" merge feature-1
run "git diff" git -C "$G/repo" diff HEAD~1
run "git diff --stat" git -C "$G/repo" diff --stat HEAD~1
run "git diff --name-only" git -C "$G/repo" diff --name-only HEAD~1
run "git show" git -C "$G/repo" show --stat
run "git show HEAD" git -C "$G/repo" show HEAD
run "git stash" echo "temp" >> "$G/repo/README.md" && git -C "$G/repo" stash
run "git stash list" git -C "$G/repo" stash list
run "git stash pop" git -C "$G/repo" stash pop
run "git tag v1.0" git -C "$G/repo" tag v1.0
run "git tag -a" git -C "$G/repo" tag -a v1.1 -m "Release 1.1"
run "git tag -l" git -C "$G/repo" tag -l
run "git describe" git -C "$G/repo" describe --tags 2>/dev/null || true
run "git reflog" git -C "$G/repo" reflog | head -5
run "git blame" git -C "$G/repo" blame README.md
run "git grep" git -C "$G/repo" grep "hello" || true
run "git rev-parse HEAD" git -C "$G/repo" rev-parse HEAD
run "git rev-list" git -C "$G/repo" rev-list --count HEAD
run "git cat-file" git -C "$G/repo" cat-file -t HEAD
run "git cat-file -p" git -C "$G/repo" cat-file -p HEAD
run "git ls-files" git -C "$G/repo" ls-files
run "git ls-tree" git -C "$G/repo" ls-tree HEAD
run "git fsck" git -C "$G/repo" fsck 2>/dev/null
run "git gc" git -C "$G/repo" gc --quiet
run "git count-objects" git -C "$G/repo" count-objects -v
run "git config --list" git -C "$G/repo" config --list
run "git init bare" git init --bare "$G/bare.git"
run "git remote add" git -C "$G/repo" remote add origin "$G/bare.git"
run "git push" git -C "$G/repo" push origin main 2>/dev/null || git -C "$G/repo" push origin master 2>/dev/null
run "git remote -v" git -C "$G/repo" remote -v
run_t 30 "git clone github" git clone --depth 1 https://github.com/octocat/Hello-World.git "$G/hello-world" 2>/dev/null
run "git cherry-pick --abort" git -C "$G/repo" cherry-pick --abort 2>/dev/null || true
run "git rebase --abort" git -C "$G/repo" rebase --abort 2>/dev/null || true
run "git clean -n" git -C "$G/repo" clean -n
run "git archive" git -C "$G/repo" archive --format=tar HEAD > "$G/archive.tar"
rm -rf "$G"
domain_end
