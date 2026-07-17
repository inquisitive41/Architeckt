# Contributing to Architeckt

First off, thank you for considering contributing to Architeckt! It's people like you that make Architeckt such a great tool.

## 1. Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check our [Issues](../../issues) to see if someone else in the community has already created a ticket. If not, go ahead and make one!

## 2. Fork & create a branch

If this is something you think you can fix, then fork Architeckt and create a branch with a descriptive name.

## 3. Get the test suite running

Make sure your environment is set up according to [INSTALLATION.md](docs/INSTALLATION.md). 
Run the tests to ensure everything is working:

```bash
export PYTHONPATH="src"
python -m pytest tests/
```

## 4. Implement your fix or feature

At this point, you're ready to make your changes. Feel free to ask for help; everyone is a beginner at first.

> [!IMPORTANT]
> - Ensure you add type hints to new functions.
> - Maintain our formatting standards (we use `black` and `flake8`).

## 5. Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with Architeckt's master branch:

```bash
git remote add upstream git@github.com:your-org/Architeckt.git
git checkout master
git pull upstream master
```

Then update your feature branch from your local copy of master, and push it! Finally, go to GitHub and make a Pull Request.
