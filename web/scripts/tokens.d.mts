type Format = (x: number) => string;

export declare const int: Format;
export declare const dp1: Format;
export declare const dp2: Format;
export declare const dp3: Format;
export declare const dp4: Format;
export declare const pct0: Format;
export declare const pct1: Format;
export declare const signed3: Format;
export declare const sig3: Format;
export declare const formats: Record<string, (x: unknown) => string>;
export declare function resolve(data: unknown, path: string): unknown;
export declare function fillTokens(html: string, data: unknown): string;
