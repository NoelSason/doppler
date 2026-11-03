"""Built-in engine-class table for ICAO aircraft type designators.

Source: ICAO Doc 8643 "Aircraft Type Designators" (the four-character codes
that ADS-B databases such as the OpenSky aircraft database and readsb's
``t`` field carry). This is a hand-curated SUBSET, not the full document.

Policy for this table: a wrong "jet" label makes ``filter_out_jets`` delete
the correct answer silently, so a designator is listed as a jet only when
its engine type is beyond doubt. Anything uncertain is left OUT (it then
reports as "unknown" and is kept). Helicopters, gliders, balloons and
tiltrotors are deliberately absent.

Values are one of ``"jet"``, ``"turboprop"``, ``"piston"``, ``"electric"``.
Use :func:`dopplergeo.adsb.aircraft_types.load_type_table` to extend or
override it from a CSV at runtime.
"""

from __future__ import annotations

_JET: tuple[str, ...] = (
    # Airbus
    "A19N", "A20N", "A21N", "A306", "A30B", "A310", "A318", "A319", "A320",
    "A321", "A332", "A333", "A337", "A338", "A339", "A342", "A343", "A345",
    "A346", "A359", "A35K", "A388", "A3ST", "BCS1", "BCS3",
    # Antonov jets
    "A124", "A148", "A158", "A225", "AN72", "AN74",
    # Boeing
    "B37M", "B38M", "B39M", "B3XM", "B703", "B712", "B720", "B721", "B722",
    "B731", "B732", "B733", "B734", "B735", "B736", "B737", "B738", "B739",
    "B741", "B742", "B743", "B744", "B748", "B74R", "B74S", "B752", "B753",
    "B762", "B763", "B764", "B772", "B773", "B77L", "B77W", "B778", "B779",
    "B788", "B789", "B78X",
    # McDonnell Douglas / Douglas jets
    "MD11", "MD80", "MD81", "MD82", "MD83", "MD87", "MD88", "MD90", "DC10",
    "DC85", "DC86", "DC87", "DC91", "DC92", "DC93", "DC94", "DC95",
    # Lockheed / Convair / Concorde airliners
    "L101", "CONC",
    # Embraer jets
    "E135", "E145", "E45X", "E170", "E175", "E75L", "E75S", "E190", "E195",
    "E290", "E295", "E35L", "E50P", "E55P", "E545", "E550",
    # Bombardier / Canadair
    "CRJ1", "CRJ2", "CRJ7", "CRJ9", "CRJX", "CL30", "CL35", "CL60", "GL5T",
    "GL7T", "GLEX",
    # Gulfstream / IAI
    "GLF2", "GLF3", "GLF4", "GLF5", "GLF6", "GA5C", "GA6C", "G150", "G280",
    "GALX", "ASTR", "WW24",
    # Cessna Citations
    "C500", "C501", "C510", "C525", "C25A", "C25B", "C25C", "C526", "C550",
    "C551", "C560", "C56X", "C650", "C680", "C68A", "C700", "C750",
    # Learjet
    "LJ23", "LJ24", "LJ25", "LJ28", "LJ31", "LJ35", "LJ36", "LJ40", "LJ45",
    "LJ55", "LJ60", "LJ70", "LJ75",
    # Dassault Falcon
    "FA10", "FA20", "FA50", "F2TH", "F900", "FA7X", "FA8X", "FA6X",
    # Hawker / HS125 / Beechjet / Premier / Sabreliner / Mitsubishi Diamond
    "H25A", "H25B", "H25C", "HA4T", "BE40", "PRM1", "SBR1", "MU30",
    # Other business / light jets
    "HDJT", "PC24", "EA50", "SF50",
    # Sukhoi / Comac / Fokker / BAe / Dornier jets
    "SU95", "C919", "ARJ2", "F70", "F100", "F28", "BA11", "B461", "B462",
    "B463", "RJ1H", "RJ85", "RJ70", "J328",
    # Tupolev / Ilyushin / Yakovlev jets
    "T134", "T154", "T204", "IL62", "IL76", "IL86", "IL96", "YK40", "YK42",
    # Military and trainer jets
    "F16", "F15", "F18", "F35", "F22", "F14", "F4", "F5", "A10", "A4", "T38",
    "T45", "T33", "C17", "K35R", "K35E", "R135", "E3TF", "E3CF", "E6", "P8",
    "B52", "B1", "B2", "C5", "C5M", "EUFI", "TOR", "HAWK", "L39", "MG29",
    "MG21", "SU27", "SU30", "SU35", "GRIP", "MIR2", "RFAL", "AJET", "M346",
    "HAR",
)

_TURBOPROP: tuple[str, ...] = (
    # ATR / Dash 8 / Saab / Embraer
    "AT43", "AT44", "AT45", "AT46", "AT72", "AT73", "AT75", "AT76", "DH8A",
    "DH8B", "DH8C", "DH8D", "SF34", "SB20", "E120", "E110", "E121",
    # Beechcraft King Air family and Beech 99 / 1900
    "BE10", "BE20", "BE30", "BE9L", "BE9T", "BE99", "B190", "B350",
    # Pilatus / Daher / Cessna / Kodiak / Epic / Piaggio / Mitsubishi
    "PC12", "PC6T", "PC7", "PC9", "PC21", "TBM7", "TBM8", "TBM9", "C208",
    "C441", "C425", "F406", "KODI", "EPIC", "P180", "MU2",
    # Piper Cheyenne and Meridian / M500 / M600 (PA-46 turboprop variants)
    "PAY1", "PAY2", "PAY3", "PAY4", "P46T",
    # Twin Commander / Swearingen Metro / Let / Dornier / BAe / Shorts
    "AC90", "AC95", "SW3", "SW4", "L410", "D228", "D328", "JS31", "JS32",
    "JS41", "ATP", "SH33", "SH36", "SC7", "N262",
    # de Havilland Canada
    "DHC6", "DHC7", "DH2T", "DH3T", "DHC5",
    # Fokker / Convair / Lockheed Electra / CASA / Harbin
    "F27", "F50", "CVLT", "L188", "C212", "CN35", "C295", "Y12",
    # Antonov / Ilyushin turboprops
    "AN12", "AN22", "AN24", "AN26", "AN28", "AN30", "AN32", "AN38", "A140",
    "IL18",
    # Military transports and trainers
    "C130", "C30J", "L100", "P3", "E2", "C2", "C27J", "G222", "A400", "TEX2",
    "TUCA", "T34T",
)

_PISTON: tuple[str, ...] = (
    # Cessna singles and twins
    "C120", "C140", "C150", "C152", "C162", "C170", "C172", "C175", "C177",
    "C180", "C182", "C185", "C188", "C190", "C195", "C205", "C206", "C207",
    "C210", "T210", "P210", "T206", "T182", "C72R", "C77R", "C82R", "C303",
    "C310", "C320", "C335", "C336", "C337", "C340", "C402", "C404", "C411",
    "C414", "C421",
    # Piper
    "J3", "PA11", "PA12", "PA14", "PA16", "PA18", "PA20", "PA22", "PA23",
    "PA24", "PA25", "PA27", "PA30", "PA31", "PA32", "PA34", "PA36", "PA38",
    "PA44", "PA46", "P28A", "P28B", "P28R", "P28S", "P28T", "P32R", "P32T",
    "AEST",
    # Beechcraft pistons
    "BE17", "BE18", "BE19", "BE23", "BE24", "BE33", "BE35", "BE36", "BE50",
    "BE55", "BE58", "BE60", "BE65", "BE76", "BE77", "BE80", "BE95",
    # Mooney / Cirrus / Diamond
    "M20P", "M20T", "SR20", "SR22", "DA20", "DA40", "DA42", "DA50", "DA62",
    "DV20",
    # Van's RV and other kit / composite singles
    "RV3", "RV4", "RV6", "RV7", "RV8", "RV9", "RV10", "RV12", "RV14", "GLST",
    "LNC2", "LNC4", "LNCE", "COL3", "COL4",
    # Grumman American / Bellanca / Citabria / Pitts / Robin / Socata
    "AA1", "AA5", "BL8", "BL17", "CH7A", "CH7B", "PTS1", "PTS2", "DR40",
    "TB9", "TB10", "TB20", "TB21",
    # Aero Commander / Navion / Grob / CAP / Extra / GippsAero / Tecnam
    "AC11", "AC50", "NAVI", "G115", "G109", "CP10", "EXTR", "GA8", "P06T",
    "P208", "ERCO", "LA4", "T34P", "YK52",
    # Vintage transports and bush planes
    "AN2", "DC3", "DC6", "CVLP", "DHC2", "DHC3", "DHC4",
)

# No electric designators are listed on purpose: the handful of certified
# electric types (e.g. Pipistrel Velis Electro) have designators the author
# could not verify. Electric classes still arrive through load_type_table
# ("L1E" in the OpenSky database).
_ELECTRIC: tuple[str, ...] = ()

TYPECODES: dict[str, str] = {}
for _cls, _codes in (
    ("jet", _JET),
    ("turboprop", _TURBOPROP),
    ("piston", _PISTON),
    ("electric", _ELECTRIC),
):
    for _code in _codes:
        if _code in TYPECODES and TYPECODES[_code] != _cls:
            raise RuntimeError(f"typecode {_code!r} listed under two classes")
        TYPECODES[_code] = _cls
del _cls, _codes, _code
