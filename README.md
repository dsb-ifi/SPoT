# Project Website

This repository updates the [jekyllized version](https://github.com/shunzh/project_website) of the source code for the [Nerfies website](https://nerfies.github.io).
You only need to change the content of [index.md](/index.md). 
It's possible to only write in markdown, but you can also use HTML to achieve fancier effects.

Here is this repository [example website](https://dsb-ifi.github.io/project-template/).

## Test it locally

Install [Jekyll](https://jekyllrb.com/docs/installation/), and run
```
jekyll serve
```
in this directory.

Then you can see the website at `http://127.0.0.1:4000`.

## To use it

- You need to copy from the `gh-pages` branch the `index.md` and workflow (`jekyll-build.yml`) files into your repository, and update the [index.md](/index.md).  For instance
  ```bash
  # create the local gh-pages
  git switch -b gh-pages

  # add the template remote
  git remote add template https://github.com/dsb-ifi/project-template.git
  # get the branch from the templates
  git checkout template/gh-pages
  # checkout the files you need
  git checkout template/gh-pages -- index.md .github/workflows/jekyll-build.yml

  # add the files to your local branch and commit
  git add index.md .github/workflows/jekyll-build.yml
  git commit index.md .github/workflows/jekyll-build.yml -m "Add template files"
  
  # remove the remote
  git remote rm template

  # Update index.md with your information
  ```
- In your repository, make sure that the `gh-pages` has permission within the "Deployment branches and tags"