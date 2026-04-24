import re
from pydantic import BaseModel, Field, field_validator


CATEGORIES: dict[str, str] = {
    "DES": "Design",
    "MAT": "Materials",
    "TST": "Testing",
    "CON": "Construction",
    "MNT": "Maintenance",
    "STD": "Standards",
    "CAS": "Case Studies",
    "RES": "Research",
    "SFT": "Safety",
    "FLX": "Flexible",
    "RGD": "Rigid",
    "CMP": "Composite",
    "OVL": "Overlay",
    "EQP": "Equipment (construction)",
    "ACT": "Activity (construction)",
}

DOCTYPE: dict[str, str] = {
    "SPEC": "Specification",
    "PROC": "Procedure",
    "GDE": "Guide",
    "MAN": "Manual",
    "RPT": "Report",
    "TAB": "Table",
    "FIG": "Figure",
    "STD": "Standard",
    "REG": "Regulation",
    "POL": "Policy",
    "GLO": "Glossary",
    "BPR": "Best Practices",
    "TLB": "Toolbox Talk",
    "PRE": "Presentation",
}

PARSERS: dict[str, str] = {
    "MDN": "marktidown",
    "DCL": "docling",
    "MRK": "marker-pdf",
    "UNS": "unstructured",
    "MUP": "pymupdf",
    "MST": "mistral"
}


class StdName(BaseModel):
    filename: str = Field(alias="filename", default="")
    ext: str = Field(alias="ext", default="md")
    id: str = Field(alias="id", default="")
    chapter: str = Field(alias="chapter", default="")
    cats: list[str] = Field(alias="cats", default=[])
    doctype: str = Field(alias="doctype", default="")
    title: str = Field(alias="title", default="")
    source: str = Field(alias="source", default="")
    year: str = Field(alias="year", default="")
    version: str = Field(alias="version", default="v01")
    parser: str = Field(alias="parser", default="")

    @field_validator("chapter")
    def val_chapter(cls, v: str) -> str:
        if v == "000":
            return "UKN000"
        return v

    @property
    def catnames(self) -> list[str]:
        return [CATEGORIES.get(cat, "UKN") for cat in self.cats]

    @property
    def doctype_name(self) -> str:
        return DOCTYPE.get(self.doctype, "UKN")

    @property
    def parser_name(self) -> str:
        return PARSERS.get(self.parser, "UKN")

    @property
    def section_no(self) -> str:
        temp_no: str = self.chapter.split("-")[0]
        temp_no = re.sub(r"[A-Z]+", "", temp_no)
        temp_no = temp_no.lstrip("0")
        return temp_no

    @property
    def subsection(self) -> str:
        return "-".join(self.chapter.split("-")[1:])


def parse_filename(filename: str) -> StdName:
    items: list[str] = filename.split("_")
    return StdName(
        **{
            "filename": filename,
            "ext": items[-1].split(".")[-1],
            "id": items[0],
            "chapter": items[1].upper(),
            "cats": items[2].split("-"),
            "doctype": items[3],
            "title": items[4],
            "source": items[5],
            "year": items[6],
            "version": items[7],
            "parser": items[-1].split(".")[0],
        }
    )
