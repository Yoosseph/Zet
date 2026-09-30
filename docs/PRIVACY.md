# Keeping local data private

Zet runs locally. Task folders default to `~/zet` (or `$ZET_ROOT`), outside the source checkout.
They can contain your questions, examples, predictions, corrections, and calibration records.
Keep `ZET_ROOT` outside any Git repository. Model downloads and benchmark source data should also
stay outside the checkout.

`.gitignore` blocks common task files, dataset formats, model weights, caches, virtual
environments, and credentials from being added accidentally. It cannot recognize every private
file, and it does not remove a file that was already committed. Before a commit or push, inspect
`git status` and `git diff --cached --name-only`; review the content of every new file. Never use
`git add -f` for real user data.

If private data was committed, stop before publishing or pushing further. Deleting the file in a
new commit does not erase it from Git history. Keep the repository private while you remove the
data from all reachable history and assess any copies already shared. Rotate any exposed secret.

The committed support-email fixture is synthetic. The MASSIVE benchmark script downloads its
source dataset outside the repository and commits only aggregate results.
