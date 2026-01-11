export const ID_TO_NAME_MAP: Record<string, string> = {
    "user_01": "Dr. Sarah Chen",
    "user_02": "Agent James Miller",
    "user_03": "Prof. Elena Rossi",
    "user_04": "Cmdr. David Okonjo",
    "user_05": "Dr. Emily Zhang",
    "user_06": "Officer Marcus Thorne",
    "user_07": "Dr. Raj Patel",
    "user_08": "Agent Sophie Dubois",
    "user_09": "Director Alan Grant",
    "user_10": "Dr. Olivia Martinez",
    "user_11": "Tech. Ryan Cooper",
    "user_12": "Spec. Lisa Wong",
    "user_13": "Dr. Thomas Anderson",
    "user_14": "Agent Natasha Romanoff",
    "user_15": "Dr. Bruce Banner",
    "user_16": "Officer John McClane",
    "user_17": "Agent Ethan Hunt",
    "user_18": "Dr. Dana Scully",
    "user_19": "Agent Fox Mulder",
    "user_20": "Cmdr. Ellen Ripley"
};

const LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"
];

const FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth",
    "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen",
    "Christopher", "Nancy", "Daniel", "Lisa", "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra"
];

const TITLES = ["Dr.", "Agent", "Prof.", "Officer", "Cmdr.", "Spec.", "Director", "Tech."];

// Deterministically generate a name for IDs not in the manual map
export function getDisplayName(id: string): string {
    if (ID_TO_NAME_MAP[id]) return ID_TO_NAME_MAP[id];

    // If no manual map, generate one deterministically based on the ID string
    let hash = 0;
    for (let i = 0; i < id.length; i++) {
        hash = id.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    const absHash = Math.abs(hash);
    const title = TITLES[absHash % TITLES.length];
    const first = FIRST_NAMES[(absHash >> 3) % FIRST_NAMES.length];
    const last = LAST_NAMES[(absHash >> 7) % LAST_NAMES.length];

    return `${title} ${first} ${last}`;
}
