"""Registry of the regulatory documents in the corpus, with version metadata.

Adding a new grid code (or a new version of one) means adding an entry here and
re-running ingestion. Retrieval can then filter by jurisdiction and date in force.
"""

DOCUMENTS = [
    {
        "doc_id": "EU-2016-631",
        "file": "rfg_2016_631_en.pdf",
        "title": "Commission Regulation (EU) 2016/631 - Network code on requirements for grid connection of generators (RfG)",
        "short": "RfG (EU) 2016/631",
        "jurisdiction": "EU",
        "language": "en",
        "version": "OJ L 112, 27.4.2016",
        "valid_from": "2016-05-17",
        "valid_to": None,
        # Official PDF from the EU Publications Office repository (EUR-Lex blocks scripted downloads)
        "url": "http://publications.europa.eu/resource/cellar/1267e3d1-0c3f-11e6-ba9a-01aa75ed71a1.0006.01/DOC_1",
    },
]


def get_document(doc_id):
    return next(d for d in DOCUMENTS if d["doc_id"] == doc_id)
