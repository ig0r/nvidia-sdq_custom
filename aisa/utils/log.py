from loguru import logger

try:
    logger.level("HEADER", icon="📢", no=17)
    logger.level("NLP", icon="🧠", color="<magenta>", no=10)
    logger.level("MODEL", icon="🤖", color="<white>", no=10)
    logger.level("COST", icon="💰", color="<green>", no=11)
    logger.level("RESP", icon="📨", color="<yellow>", no=12)
    logger.level("TIME", icon="⏱", color="<cyan>", no=13)
    logger.level("CHUNK", icon="📦", color="<white>", no=14)
except ValueError:
    pass  # levels already exist


def heading(heading: str) -> None:
    logger.opt(colors=True).log("HEADER", "<black>" + "=" * 50 + "</black>")
    logger.opt(colors=True).log("HEADER", f"{heading.upper()}")
    logger.opt(colors=True).log("HEADER", "<black>" + "=" * 50 + "</black>")
