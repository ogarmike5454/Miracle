"""Rule set used for linguistic defect indicators (unchanged from the original notebook)."""
TAXONOMY_RULES = {
    'Configuration/environment': [
        r'\bconfig(?:uration)?\b', r'\benvironment\b', r'\benv\b', r'\bsetting(?:s)?\b',
        r'\bflag(?:s)?\b', r'\byaml\b', r'\bjson\b', r'\bpermission(?:s)?\b'
    ],
    'Interface/API': [
        r'\bapi\b', r'\binterface\b', r'\bendpoint\b', r'\brequest\b', r'\bresponse\b',
        r'\bhttp\b', r'\bgrpc\b', r'\bclient\b', r'\bserver\b'
    ],
    'Logic/control flow': [
        r'\blogic\b', r'\bcondition\b', r'\bbranch\b', r'\bincorrect\b', r'\bwrong\b',
        r'\bexception\b', r'\bpanic\b', r'\bnull\b', r'\bnil\b', r'\berror\b'
    ],
    'Data/state': [
        r'\bdata\b', r'\bstate\b', r'\bvalue\b', r'\bfield\b', r'\bschema\b',
        r'\bserialization\b', r'\bdeserialize\w*\b', r'\bdatabase\b', r'\bcache\b'
    ],
    'Concurrency/timing': [
        r'\brace condition\b', r'\bconcurren\w*\b', r'\bdeadlock\b', r'\block(?:ing)?\b',
        r'\bthread\b', r'\btimeout\b', r'\btiming\b', r'\basynchronous\b', r'\basync\b'
    ],
    'Dependency/build': [
        r'\bdependency\b', r'\bdependencies\b', r'\bversion\b', r'\bupgrade\b', r'\bdowngrade\b',
        r'\bbuild\b', r'\bcompile\w*\b', r'\bpackage\b', r'\bmodule\b', r'\blibrary\b'
    ],
    'Performance/resource': [
        r'\bperformance\b', r'\bmemory\b', r'\bleak\b', r'\bcpu\b', r'\bresource\b',
        r'\blatency\b', r'\bslow\b', r'\boverflow\b', r'\bout of memory\b', r'\boom\b'
    ],
    'Security/access': [
        r'\bsecurity\b', r'\bauthentication\b', r'\bauthorization\b', r'\bauthorisation\b',
        r'\bcredential\w*\b', r'\btoken\b', r'\baccess denied\b', r'\bunauthori[sz]ed\b'
    ],
}

CAUSAL_PATTERNS = [
    r'\bbecause of\b', r'\bbecause\b', r'\bdue to\b', r'\bcaused by\b',
    r'\bresults? from\b', r'\btriggered by\b', r'\bleads? to\b', r'\broot cause\b'
]
