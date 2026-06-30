.PHONY: pdf html crossref production clean

PAPER := paper.md
JOURNAL := joss
INARA_IMAGE := openjournals/inara
DOCKER_RUN := docker run --rm -it \
	-e JOURNAL=$(JOURNAL) \
	-v $(CURDIR):/data \
	-u $(shell id -u):$(shell id -g) \
	$(INARA_IMAGE)

pdf:
	$(DOCKER_RUN) -o pdf $(PAPER)

html:
	$(DOCKER_RUN) -o html $(PAPER)

crossref:
	$(DOCKER_RUN) -o crossref $(PAPER)

production:
	$(DOCKER_RUN) -p -o pdf $(PAPER)

clean:
	rm -f paper.pdf paper.html paper.crossref
