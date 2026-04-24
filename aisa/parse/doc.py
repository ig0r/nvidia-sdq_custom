import pandas as pd
from aisa.utils import files, logger
from aisa.parse.naming import StdName, parse_filename


META_EXT: str = "-metadata.csv"


class Metadata:
    def __init__(self, meta_dir: str):
        self.dfs: dict[str, list[dict[str, any]]] = {}
        main_path: str = files.os_path(f"{meta_dir}/_main.csv")
        self.main: list[dict[str, any]] = (
            pd.read_csv(main_path).fillna("").to_dict(orient="records")
        )
        self.curr_df: list[dict[str, any]] = []
        for f in files.find_files(meta_dir, META_EXT):
            main_name: str = files.split_path(f)[1].replace(META_EXT, "")
            if main_name not in self.dfs:
                self.dfs[main_name] = pd.read_csv(f).to_dict(orient="records")

    def get_section_name(self, docid: str, sect_no: str) -> str:
        if docid not in list(self.dfs.keys()):
            self.curr_df = []
            return "Unknown"

        self.curr_df: list[dict[str, any]] = self.dfs[docid]
        return str(
            [entry for entry in self.curr_df if str(entry["section_no"]) == sect_no][0][
                "section_title"
            ]
        ).title()

    def get_subsection_name(self, docid: str, sect_no: str, sub_no: str) -> str:
        if self.curr_df == []:
            return "Unknown"
        return str(
            [
                entry
                for entry in self.curr_df
                if str(entry["section_no"]) == sect_no
                and str(entry["subsection"]) == sub_no
            ][0]["title"]
        ).title()

    def list_subsections(self, docid: str, sect_no: str) -> dict[str, str]:
        if self.curr_df == []:
            return {}

        new_dict: dict[str, str] = {}
        for entry in [
            item for item in self.curr_df if str(item["section_no"]) == sect_no
        ]:
            new_dict[str(entry["subsection"])] = str(entry["title"]).title()
        return new_dict


class ParsedDoc:
    def __init__(self, filepath: str, chunk_dir: str, gen_meta: dict = {}):
        abs_path, filename = files.split_path(filepath)
        self.doctitle: str = gen_meta.get("doctitle", "Unknown")
        self.pub_no: str = gen_meta.get("pub_no", "")
        self.parent_dir: str = abs_path
        self.name: StdName = parse_filename(filename)
        self.year: str = self.name.year
        self.section_no: str = self.name.section_no
        self.section_name: str = "Unknown"
        self.subsections: dict[str, str] = (
            {self.name.subsection: "Unknown"} if self.name.subsection != "" else {}
        )
        self.all_subs: dict[str, str] = {}
        self.content: str = ""
        self.pdf_path: str = ""
        self.base_out: str = files.os_path(
            f"{chunk_dir}/{self.name.id}_{self.name.chapter}"
        )
        try:
            self.content = files.read_text_file(filepath)
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            exit()

    def __repr__(self):
        return (
            f"ParsedDoc(name={self.doctitle}, "
            + f"section_name={self.section_name}, "
            + f"section_no={self.section_no})"
        )

    def assign_names(self, metadata: Metadata) -> None:
        self.section_name = metadata.get_section_name(self.name.id, self.section_no)
        self.all_subs = metadata.list_subsections(self.name.id, self.section_no)
        main_entry: dict = [
            entry
            for entry in metadata.main
            if entry["docid"] == self.name.id and entry["chapter"] == self.name.chapter
        ][0]
        self.pdf_path = (
            main_entry["pdfpath"]
            if main_entry["pdfpath"] != ""
            else main_entry["og_pdfpath"]
        )
        if self.subsections == {}:
            self.subsections = self.all_subs
        else:
            self.subsections[self.name.subsection] = metadata.get_subsection_name(
                self.name.id, self.section_no, self.name.subsection
            )
